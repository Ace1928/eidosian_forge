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
class Router:

    def __init__(self, controller_handle: ActorHandle, deployment_id: DeploymentID, self_node_id: str, self_actor_id: str, self_availability_zone: Optional[str], event_loop: asyncio.BaseEventLoop=None, _prefer_local_node_routing: bool=False, _router_cls: Optional[str]=None):
        """Used to assign requests to downstream replicas for a deployment.

        The scheduling behavior is delegated to a ReplicaScheduler; this is a thin
        wrapper that adds metrics and logging.
        """
        self._event_loop = event_loop
        self.deployment_id = deployment_id
        if _router_cls:
            self._replica_scheduler = load_class(_router_cls)(event_loop=event_loop, deployment_id=deployment_id)
        else:
            self._replica_scheduler = PowerOfTwoChoicesReplicaScheduler(event_loop, deployment_id, _prefer_local_node_routing, RAY_SERVE_PROXY_PREFER_LOCAL_AZ_ROUTING, self_node_id, self_actor_id, self_availability_zone)
        logger.info(f'Using router {self._replica_scheduler.__class__}.', extra={'log_to_stderr': False})
        self.num_router_requests = metrics.Counter('serve_num_router_requests', description='The number of requests processed by the router.', tag_keys=('deployment', 'route', 'application'))
        self.num_router_requests.set_default_tags({'deployment': deployment_id.name, 'application': deployment_id.app})
        self.num_queued_queries = 0
        self.num_queued_queries_gauge = metrics.Gauge('serve_deployment_queued_queries', description='The current number of queries to this deployment waiting to be assigned to a replica.', tag_keys=('deployment', 'application'))
        self.num_queued_queries_gauge.set_default_tags({'deployment': deployment_id.name, 'application': deployment_id.app})
        self.long_poll_client = LongPollClient(controller_handle, {(LongPollNamespace.RUNNING_REPLICAS, deployment_id): self._replica_scheduler.update_running_replicas}, call_in_event_loop=event_loop)
        deployment_route = DeploymentRoute.FromString(ray.get(controller_handle.get_deployment_info.remote(*deployment_id)))
        deployment_info = DeploymentInfo.from_proto(deployment_route.deployment_info)
        self.metrics_pusher = None
        if deployment_info.deployment_config.autoscaling_config:
            self.autoscaling_enabled = True
            self.push_metrics_to_controller = controller_handle.record_handle_metrics.remote
            self.metrics_pusher = MetricsPusher()
            self.metrics_pusher.register_task(self._collect_handle_queue_metrics, HANDLE_METRIC_PUSH_INTERVAL_S, self.push_metrics_to_controller)
            self.metrics_pusher.start()
        else:
            self.autoscaling_enabled = False

    def _collect_handle_queue_metrics(self) -> Dict[str, int]:
        return {self.deployment_id: self.num_queued_queries}

    async def assign_request(self, request_meta: RequestMetadata, *request_args, **request_kwargs) -> Union[ray.ObjectRef, 'ray._raylet.ObjectRefGenerator']:
        """Assign a query to a replica and return the resulting object_ref."""
        self.num_router_requests.inc(tags={'route': request_meta.route})
        self.num_queued_queries += 1
        self.num_queued_queries_gauge.set(self.num_queued_queries)
        if self.autoscaling_enabled and len(self._replica_scheduler.curr_replicas) == 0 and (self.num_queued_queries == 1):
            self.push_metrics_to_controller({self.deployment_id: 1}, time.time())
        try:
            query = Query(args=list(request_args), kwargs=request_kwargs, metadata=request_meta)
            await query.replace_known_types_in_args()
            return await self._replica_scheduler.assign_replica(query)
        finally:
            self.num_queued_queries -= 1
            self.num_queued_queries_gauge.set(self.num_queued_queries)

    def shutdown(self):
        """Shutdown router gracefully.

        The metrics_pusher needs to be shutdown separately.
        """
        if self.metrics_pusher:
            self.metrics_pusher.shutdown()