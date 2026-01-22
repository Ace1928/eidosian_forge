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
def _stop_replicas_on_draining_nodes(self):
    draining_nodes = self._cluster_node_info_cache.get_draining_node_ids()
    for replica in self._replicas.pop(states=[ReplicaState.UPDATING, ReplicaState.RUNNING]):
        if replica.actor_node_id in draining_nodes:
            state = replica._actor_details.state
            logger.info(f"Stopping replica {replica.replica_tag} (currently {state}) of deployment '{self.deployment_name}' in application '{self.app_name}' on draining node {replica.actor_node_id}.")
            self._stop_replica(replica, graceful_stop=True)
        else:
            self._replicas.add(replica.actor_details.state, replica)