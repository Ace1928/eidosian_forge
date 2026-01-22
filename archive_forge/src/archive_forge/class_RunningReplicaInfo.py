import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import List, NamedTuple, Optional
from ray.actor import ActorHandle
from ray.serve.generated.serve_pb2 import ApplicationStatus as ApplicationStatusProto
from ray.serve.generated.serve_pb2 import (
from ray.serve.generated.serve_pb2 import DeploymentStatus as DeploymentStatusProto
from ray.serve.generated.serve_pb2 import (
from ray.serve.generated.serve_pb2 import (
from ray.serve.generated.serve_pb2 import (
from ray.serve.generated.serve_pb2 import StatusOverview as StatusOverviewProto
@dataclass(frozen=True)
class RunningReplicaInfo:
    deployment_name: str
    replica_tag: ReplicaTag
    node_id: Optional[str]
    availability_zone: Optional[str]
    actor_handle: ActorHandle
    max_concurrent_queries: int
    is_cross_language: bool = False
    multiplexed_model_ids: List[str] = field(default_factory=list)

    def __post_init__(self):
        hash_val = hash(' '.join([self.deployment_name, self.replica_tag, self.node_id if self.node_id else '', str(self.actor_handle._actor_id), str(self.max_concurrent_queries), str(self.is_cross_language), str(self.multiplexed_model_ids)]))
        object.__setattr__(self, '_hash', hash_val)

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        return all([isinstance(other, RunningReplicaInfo), self._hash == other._hash])