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
def _detect_and_remove_leaked_placement_groups(self, all_current_actor_names: List[str], all_current_placement_group_names: List[str]):
    """Detect and remove any placement groups not associated with a replica.

        This can happen under certain rare circumstances:
            - The controller creates a placement group then crashes before creating
            the associated replica actor.
            - While the controller is down, a replica actor crashes but its placement
            group still exists.

        In both of these (or any other unknown cases), we simply need to remove the
        leaked placement groups.
        """
    leaked_pg_names = []
    for pg_name in all_current_placement_group_names:
        if ReplicaName.is_replica_name(pg_name) and pg_name not in all_current_actor_names:
            leaked_pg_names.append(pg_name)
    if len(leaked_pg_names) > 0:
        logger.warning(f'Detected leaked placement groups: {leaked_pg_names}. The placement groups will be removed. This can happen in rare circumstances when the controller crashes and should not cause any issues. If this happens repeatedly, please file an issue on GitHub.')
    for leaked_pg_name in leaked_pg_names:
        try:
            pg = ray.util.get_placement_group(leaked_pg_name)
            ray.util.remove_placement_group(pg)
        except Exception:
            logger.exception(f'Failed to remove leaked placement group {leaked_pg_name}.')