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
def _send_query_java(self, query: Query) -> ray.ObjectRef:
    """Send the query to a Java replica.

        Does not currently support streaming.
        """
    if query.metadata.is_streaming:
        raise RuntimeError('Streaming not supported for Java.')
    if len(query.args) != 1:
        raise ValueError('Java handle calls only support a single argument.')
    return self._actor_handle.handle_request.remote(RequestMetadataProto(request_id=query.metadata.request_id, endpoint=query.metadata.endpoint, call_method='call' if query.metadata.call_method == '__call__' else query.metadata.call_method).SerializeToString(), query.args)