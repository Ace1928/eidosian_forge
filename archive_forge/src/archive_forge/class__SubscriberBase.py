import asyncio
from collections import deque
import logging
import random
from typing import Tuple, List
import grpc
from ray._private.utils import get_or_create_event_loop
import ray._private.gcs_utils as gcs_utils
import ray._private.logging_utils as logging_utils
from ray.core.generated.gcs_pb2 import ErrorTableData
from ray.core.generated import dependency_pb2
from ray.core.generated import gcs_service_pb2_grpc
from ray.core.generated import gcs_service_pb2
from ray.core.generated import common_pb2
from ray.core.generated import pubsub_pb2
class _SubscriberBase:

    def __init__(self, worker_id: bytes=None):
        self._worker_id = worker_id
        self._subscriber_id = bytes(bytearray((random.getrandbits(8) for _ in range(28))))
        self._last_batch_size = 0
        self._max_processed_sequence_id = 0
        self._publisher_id = b''

    @property
    def last_batch_size(self):
        return self._last_batch_size

    def _subscribe_request(self, channel):
        cmd = pubsub_pb2.Command(channel_type=channel, subscribe_message={})
        req = gcs_service_pb2.GcsSubscriberCommandBatchRequest(subscriber_id=self._subscriber_id, sender_id=self._worker_id, commands=[cmd])
        return req

    def _poll_request(self):
        return gcs_service_pb2.GcsSubscriberPollRequest(subscriber_id=self._subscriber_id, max_processed_sequence_id=self._max_processed_sequence_id, publisher_id=self._publisher_id)

    def _unsubscribe_request(self, channels):
        req = gcs_service_pb2.GcsSubscriberCommandBatchRequest(subscriber_id=self._subscriber_id, sender_id=self._worker_id, commands=[])
        for channel in channels:
            req.commands.append(pubsub_pb2.Command(channel_type=channel, unsubscribe_message={}))
        return req

    @staticmethod
    def _should_terminate_polling(e: grpc.RpcError) -> None:
        if e.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
            return True
        if e.code() == grpc.StatusCode.UNAVAILABLE:
            return True
        return False

    @staticmethod
    def _pop_error_info(queue):
        if len(queue) == 0:
            return (None, None)
        msg = queue.popleft()
        return (msg.key_id, msg.error_info_message)

    @staticmethod
    def _pop_log_batch(queue):
        if len(queue) == 0:
            return None
        msg = queue.popleft()
        return logging_utils.log_batch_proto_to_dict(msg.log_batch_message)

    @staticmethod
    def _pop_function_key(queue):
        if len(queue) == 0:
            return None
        msg = queue.popleft()
        return msg.python_function_message.key

    @staticmethod
    def _pop_resource_usage(queue):
        if len(queue) == 0:
            return (None, None)
        msg = queue.popleft()
        return (msg.key_id.decode(), msg.node_resource_usage_message.json)

    @staticmethod
    def _pop_actors(queue, batch_size=100):
        if len(queue) == 0:
            return []
        popped = 0
        msgs = []
        while len(queue) > 0 and popped < batch_size:
            msg = queue.popleft()
            msgs.append((msg.key_id, msg.actor_message))
            popped += 1
        return msgs