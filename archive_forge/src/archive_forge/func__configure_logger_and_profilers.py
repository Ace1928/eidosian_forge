import asyncio
import inspect
import logging
import os
import pickle
import time
import traceback
from contextlib import asynccontextmanager
from importlib import import_module
from typing import Any, AsyncGenerator, Callable, Dict, Optional, Tuple
import aiorwlock
import starlette.responses
from starlette.requests import Request
from starlette.types import Message, Receive, Scope, Send
import ray
from ray import cloudpickle
from ray._private.async_compat import sync_to_async
from ray._private.utils import get_or_create_event_loop
from ray.actor import ActorClass, ActorHandle
from ray.remote_function import RemoteFunction
from ray.serve import metrics
from ray.serve._private.autoscaling_metrics import InMemoryMetricsStore
from ray.serve._private.common import (
from ray.serve._private.config import DeploymentConfig
from ray.serve._private.constants import (
from ray.serve._private.deployment_info import CONTROL_PLANE_CONCURRENCY_GROUP
from ray.serve._private.http_util import (
from ray.serve._private.logging_utils import (
from ray.serve._private.router import RequestMetadata
from ray.serve._private.utils import (
from ray.serve._private.version import DeploymentVersion
from ray.serve.deployment import Deployment
from ray.serve.exceptions import RayServeException
from ray.serve.grpc_util import RayServegRPCContext
from ray.serve.schema import LoggingConfig
def _configure_logger_and_profilers(self, replica_tag: ReplicaTag, logging_config: LoggingConfig):
    replica_name = ReplicaName.from_replica_tag(replica_tag)
    if replica_name.app_name:
        component_name = f'{replica_name.app_name}_{replica_name.deployment_name}'
    else:
        component_name = f'{replica_name.deployment_name}'
    component_id = replica_name.replica_suffix
    configure_component_logger(component_type=ServeComponentType.REPLICA, component_name=component_name, component_id=component_id, logging_config=logging_config)
    configure_component_memory_profiler(component_type=ServeComponentType.REPLICA, component_name=component_name, component_id=component_id)
    self.cpu_profiler, self.cpu_profiler_log = configure_component_cpu_profiler(component_type=ServeComponentType.REPLICA, component_name=component_name, component_id=component_id)