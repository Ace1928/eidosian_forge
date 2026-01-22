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
def _check_and_update_replicas(self):
    """
        Check current state of all DeploymentReplica being tracked, and compare
        with state container from previous update() cycle to see if any state
        transition happened.
        """
    for replica in self._replicas.pop(states=[ReplicaState.RUNNING]):
        if replica.check_health():
            self._replicas.add(ReplicaState.RUNNING, replica)
            self.health_check_gauge.set(1, tags={'deployment': self.deployment_name, 'replica': replica.replica_tag, 'application': self.app_name})
        else:
            logger.warning(f"Replica {replica.replica_tag} of deployment {self.deployment_name} in application '{self.app_name}' failed health check, stopping it.")
            self.health_check_gauge.set(0, tags={'deployment': self.deployment_name, 'replica': replica.replica_tag, 'application': self.app_name})
            self._stop_replica(replica, graceful_stop=not self.FORCE_STOP_UNHEALTHY_REPLICAS)
            if replica.version == self._target_state.version:
                self._curr_status_info = self._curr_status_info.update(status=DeploymentStatus.UNHEALTHY, status_trigger=DeploymentStatusTrigger.HEALTH_CHECK_FAILED, message="A replica's health check failed. This deployment will be UNHEALTHY until the replica recovers or a new deploy happens.")
    slow_start_replicas = []
    slow_start = self._check_startup_replicas(ReplicaState.STARTING)
    slow_update = self._check_startup_replicas(ReplicaState.UPDATING)
    slow_recover = self._check_startup_replicas(ReplicaState.RECOVERING, stop_on_slow=True)
    slow_start_replicas = slow_start + slow_update + slow_recover
    if len(slow_start_replicas) and time.time() - self._prev_startup_warning > SLOW_STARTUP_WARNING_PERIOD_S:
        pending_allocation = []
        pending_initialization = []
        for replica, startup_status in slow_start_replicas:
            if startup_status == ReplicaStartupStatus.PENDING_ALLOCATION:
                pending_allocation.append(replica)
            if startup_status == ReplicaStartupStatus.PENDING_INITIALIZATION:
                pending_initialization.append(replica)
        if len(pending_allocation) > 0:
            required, available = pending_allocation[0].resource_requirements()
            message = f"Deployment '{self.deployment_name}' in application '{self.app_name}' {len(pending_allocation)} replicas that have taken more than {SLOW_STARTUP_WARNING_S}s to be scheduled. This may be due to waiting for the cluster to auto-scale or for a runtime environment to be installed. Resources required for each replica: {required}, total resources available: {available}. Use `ray status` for more details."
            logger.warning(message)
            if _SCALING_LOG_ENABLED:
                print_verbose_scaling_log()
            if self._curr_status_info.status != DeploymentStatus.UNHEALTHY:
                self._curr_status_info = self._curr_status_info.update(message=message)
        if len(pending_initialization) > 0:
            message = f"Deployment '{self.deployment_name}' in application '{self.app_name}' has {len(pending_initialization)} replicas that have taken more than {SLOW_STARTUP_WARNING_S}s to initialize. This may be caused by a slow __init__ or reconfigure method."
            logger.warning(message)
            if self._curr_status_info.status != DeploymentStatus.UNHEALTHY:
                self._curr_status_info = self._curr_status_info.update(message=message)
        self._prev_startup_warning = time.time()
    for replica in self._replicas.pop(states=[ReplicaState.STOPPING]):
        stopped = replica.check_stopped()
        if not stopped:
            self._replicas.add(ReplicaState.STOPPING, replica)
        else:
            logger.info(f'Replica {replica.replica_tag} is stopped.')
            if replica.replica_tag in self.replica_average_ongoing_requests:
                del self.replica_average_ongoing_requests[replica.replica_tag]