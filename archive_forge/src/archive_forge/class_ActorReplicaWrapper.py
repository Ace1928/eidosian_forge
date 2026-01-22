import asyncio
import enum
import logging
import math
import pickle
import random
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import (
import ray
from ray._private.utils import load_class
from ray.actor import ActorHandle
from ray.dag.py_obj_scanner import _PyObjScanner
from ray.exceptions import RayActorError
from ray.serve._private.common import DeploymentID, RequestProtocol, RunningReplicaInfo
from ray.serve._private.constants import (
from ray.serve._private.deployment_info import DeploymentInfo
from ray.serve._private.long_poll import LongPollClient, LongPollNamespace
from ray.serve._private.utils import JavaActorHandleProxy, MetricsPusher
from ray.serve.generated.serve_pb2 import DeploymentRoute
from ray.serve.generated.serve_pb2 import RequestMetadata as RequestMetadataProto
from ray.serve.grpc_util import RayServegRPCContext
from ray.util import metrics
class ActorReplicaWrapper:

    def __init__(self, replica_info: RunningReplicaInfo):
        self._replica_info = replica_info
        self._multiplexed_model_ids = set(replica_info.multiplexed_model_ids)
        if replica_info.is_cross_language:
            self._actor_handle = JavaActorHandleProxy(replica_info.actor_handle)
        else:
            self._actor_handle = replica_info.actor_handle

    @property
    def replica_id(self) -> str:
        return self._replica_info.replica_tag

    @property
    def node_id(self) -> str:
        return self._replica_info.node_id

    @property
    def availability_zone(self) -> Optional[str]:
        return self._replica_info.availability_zone

    @property
    def multiplexed_model_ids(self) -> Set[str]:
        return self._multiplexed_model_ids

    async def get_queue_state(self, *, deadline_s: float) -> Tuple[int, bool]:
        obj_ref = self._actor_handle.get_num_ongoing_requests.remote()
        try:
            queue_len = await obj_ref
            accepted = queue_len < self._replica_info.max_concurrent_queries
            return (queue_len, accepted)
        except asyncio.CancelledError:
            ray.cancel(obj_ref)
            raise

    def _send_query_java(self, query: Query) -> ray.ObjectRef:
        """Send the query to a Java replica.

        Does not currently support streaming.
        """
        if query.metadata.is_streaming:
            raise RuntimeError('Streaming not supported for Java.')
        if len(query.args) != 1:
            raise ValueError('Java handle calls only support a single argument.')
        return self._actor_handle.handle_request.remote(RequestMetadataProto(request_id=query.metadata.request_id, endpoint=query.metadata.endpoint, call_method='call' if query.metadata.call_method == '__call__' else query.metadata.call_method).SerializeToString(), query.args)

    def _send_query_python(self, query: Query) -> Union[ray.ObjectRef, 'ray._raylet.ObjectRefGenerator']:
        """Send the query to a Python replica."""
        if query.metadata.is_streaming:
            method = self._actor_handle.handle_request_streaming.options(num_returns='streaming')
        else:
            method = self._actor_handle.handle_request
        return method.remote(pickle.dumps(query.metadata), *query.args, **query.kwargs)

    def send_query(self, query: Query) -> Union[ray.ObjectRef, 'ray._raylet.ObjectRefGenerator']:
        if self._replica_info.is_cross_language:
            return self._send_query_java(query)
        else:
            return self._send_query_python(query)