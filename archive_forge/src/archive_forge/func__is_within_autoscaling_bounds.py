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
def _is_within_autoscaling_bounds(self) -> bool:
    """Whether or not this deployment is within the autoscaling bounds.

        This method should only be used for autoscaling deployments. It raises
        an assertion error otherwise.

        Returns: True if the number of running replicas for the current
            deployment version is within the autoscaling bounds. False
            otherwise.
        """
    target_version = self._target_state.version
    num_replicas_running_at_target_version = self._replicas.count(states=[ReplicaState.RUNNING], version=target_version)
    autoscaling_policy = self._target_state.info.autoscaling_policy
    assert autoscaling_policy is not None
    lower_bound = autoscaling_policy.get_current_lower_bound(self._target_state.info.target_capacity, self._target_state.info.target_capacity_direction)
    upper_bound = get_capacity_adjusted_num_replicas(autoscaling_policy.config.max_replicas, self._target_state.info.target_capacity)
    return lower_bound <= num_replicas_running_at_target_version <= upper_bound