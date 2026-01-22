import json
import logging
import math
import os
import random
import time
import traceback
from collections import defaultdict
from copy import copy
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import ray
from ray import ObjectRef, cloudpickle
from ray.actor import ActorHandle
from ray.exceptions import RayActorError, RayError, RayTaskError, RuntimeEnvSetupError
from ray.serve import metrics
from ray.serve._private import default_impl
from ray.serve._private.autoscaling_metrics import InMemoryMetricsStore
from ray.serve._private.cluster_node_info_cache import ClusterNodeInfoCache
from ray.serve._private.common import (
from ray.serve._private.config import DeploymentConfig
from ray.serve._private.constants import (
from ray.serve._private.deployment_info import DeploymentInfo
from ray.serve._private.deployment_scheduler import (
from ray.serve._private.long_poll import LongPollHost, LongPollNamespace
from ray.serve._private.storage.kv_store import KVStoreBase
from ray.serve._private.usage import ServeUsageTag
from ray.serve._private.utils import (
from ray.serve._private.version import DeploymentVersion, VersionedReplica
from ray.serve.generated.serve_pb2 import DeploymentLanguage
from ray.serve.schema import (
from ray.util.placement_group import PlacementGroup
def _check_active_health_check(self) -> ReplicaHealthCheckResponse:
    """Check the active health check (if any).

        self._health_check_ref will be reset to `None` when the active health
        check is deemed to have succeeded or failed. This method *does not*
        start a new health check, that's up to the caller.

        Returns:
            - NONE if there's no active health check, or it hasn't returned
              yet and the timeout is not up.
            - SUCCEEDED if the active health check succeeded.
            - APP_FAILURE if the active health check failed (or didn't return
              before the timeout).
            - ACTOR_CRASHED if the underlying actor crashed.
        """
    if self._health_check_ref is None:
        response = ReplicaHealthCheckResponse.NONE
    elif check_obj_ref_ready_nowait(self._health_check_ref):
        try:
            ray.get(self._health_check_ref)
            response = ReplicaHealthCheckResponse.SUCCEEDED
        except RayActorError:
            response = ReplicaHealthCheckResponse.ACTOR_CRASHED
        except RayError as e:
            logger.warning(f'Health check for replica {self._replica_tag} failed: {e}')
            response = ReplicaHealthCheckResponse.APP_FAILURE
    elif time.time() - self._last_health_check_time > self.health_check_timeout_s:
        logger.warning(f"Didn't receive health check response for replica {self._replica_tag} after {self.health_check_timeout_s}s, marking it unhealthy.")
        response = ReplicaHealthCheckResponse.APP_FAILURE
    else:
        response = ReplicaHealthCheckResponse.NONE
    if response is not ReplicaHealthCheckResponse.NONE:
        self._health_check_ref = None
    return response