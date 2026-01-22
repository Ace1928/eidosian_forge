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
@dataclass
class DeploymentTargetState:
    """The current goal state for a deployment.

    info: contains the information needed to initialize a replica.
    target_num_replicas: the number of replicas to run. This should already
        be adjusted by the target_capacity.
    version: the goal version of the deployment.
    deleting: whether the deployment is being deleted.
    """
    info: Optional[DeploymentInfo]
    target_num_replicas: int
    version: Optional[DeploymentVersion]
    deleting: bool

    @classmethod
    def default(cls) -> 'DeploymentTargetState':
        return cls(None, -1, None, False)

    @classmethod
    def create(cls, info: DeploymentInfo, target_num_replicas: int, *, deleting: bool=False) -> 'DeploymentTargetState':
        if deleting:
            if target_num_replicas != 0:
                raise ValueError(f'target_num_replicas must be 0 when setting target state to deleting. Got {target_num_replicas} instead.')
        version = DeploymentVersion(info.version, deployment_config=info.deployment_config, ray_actor_options=info.replica_config.ray_actor_options, placement_group_bundles=info.replica_config.placement_group_bundles, placement_group_strategy=info.replica_config.placement_group_strategy, max_replicas_per_node=info.replica_config.max_replicas_per_node)
        return cls(info, target_num_replicas, version, deleting)

    def is_scaled_copy_of(self, other_target_state: 'DeploymentTargetState') -> bool:
        """Checks if this target state is a scaled copy of another target state.

        A target state is a scaled copy of another target state if all
        configurable info is identical, other than target_num_replicas.

        Returns: True if this target state contains a non-None DeploymentInfo
            and is a scaled copy of the other target state.
        """
        if other_target_state.info is None:
            return False
        return all([self.info.replica_config.ray_actor_options == other_target_state.info.replica_config.ray_actor_options, self.info.replica_config.placement_group_bundles == other_target_state.info.replica_config.placement_group_bundles, self.info.replica_config.placement_group_strategy == other_target_state.info.replica_config.placement_group_strategy, self.info.replica_config.max_replicas_per_node == other_target_state.info.replica_config.max_replicas_per_node, self.info.deployment_config.dict(exclude={'num_replicas'}) == other_target_state.info.deployment_config.dict(exclude={'num_replicas'}), self.version, self.version == other_target_state.version])