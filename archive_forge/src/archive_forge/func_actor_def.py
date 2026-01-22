from typing import Any, Dict, Optional
import ray
from ray.serve._private.autoscaling_policy import BasicAutoscalingPolicy
from ray.serve._private.common import TargetCapacityDirection
from ray.serve._private.config import DeploymentConfig, ReplicaConfig
from ray.serve.generated.serve_pb2 import DeploymentInfo as DeploymentInfoProto
from ray.serve.generated.serve_pb2 import (
@property
def actor_def(self):
    from ray.serve._private.replica import create_replica_wrapper
    if self._cached_actor_def is None:
        assert self.actor_name is not None
        self._cached_actor_def = ray.remote(**REPLICA_DEFAULT_ACTOR_OPTIONS)(create_replica_wrapper(self.actor_name))
    return self._cached_actor_def