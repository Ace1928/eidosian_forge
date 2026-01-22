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
class GcsAioPublisher(_PublisherBase):
    """Publisher to GCS. Uses async io."""

    def __init__(self, address: str=None, channel: aiogrpc.Channel=None):
        if address:
            assert channel is None, 'address and channel cannot both be specified'
            channel = gcs_utils.create_gcs_channel(address, aio=True)
        else:
            assert channel is not None, 'One of address and channel must be specified'
        self._stub = gcs_service_pb2_grpc.InternalPubSubGcsServiceStub(channel)

    async def publish_error(self, key_id: bytes, error_info: ErrorTableData) -> None:
        """Publishes error info to GCS."""
        msg = pubsub_pb2.PubMessage(channel_type=pubsub_pb2.RAY_ERROR_INFO_CHANNEL, key_id=key_id, error_info_message=error_info)
        req = gcs_service_pb2.GcsPublishRequest(pub_messages=[msg])
        await self._stub.GcsPublish(req)

    async def publish_logs(self, log_batch: dict) -> None:
        """Publishes logs to GCS."""
        req = self._create_log_request(log_batch)
        await self._stub.GcsPublish(req)

    async def publish_resource_usage(self, key: str, json: str) -> None:
        """Publishes logs to GCS."""
        req = self._create_node_resource_usage_request(key, json)
        await self._stub.GcsPublish(req)