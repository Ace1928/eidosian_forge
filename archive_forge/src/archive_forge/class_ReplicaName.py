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
@dataclass
class ReplicaName:
    app_name: str
    deployment_name: str
    replica_suffix: str
    replica_tag: ReplicaTag = ''
    delimiter: str = '#'
    prefix: str = 'SERVE_REPLICA::'

    def __init__(self, app_name: str, deployment_name: str, replica_suffix: str):
        self.app_name = app_name
        self.deployment_name = deployment_name
        self.replica_suffix = replica_suffix
        if app_name:
            self.replica_tag = self.delimiter.join([app_name, deployment_name, replica_suffix])
        else:
            self.replica_tag = self.delimiter.join([deployment_name, replica_suffix])

    @property
    def deployment_id(self):
        return DeploymentID(self.deployment_name, self.app_name)

    @staticmethod
    def is_replica_name(actor_name: str) -> bool:
        return actor_name.startswith(ReplicaName.prefix)

    @classmethod
    def from_str(cls, actor_name):
        assert ReplicaName.is_replica_name(actor_name)
        replica_tag = actor_name.replace(cls.prefix, '')
        return ReplicaName.from_replica_tag(replica_tag)

    @classmethod
    def from_replica_tag(cls, tag):
        parsed = tag.split(cls.delimiter)
        if len(parsed) == 3:
            return cls(app_name=parsed[0], deployment_name=parsed[1], replica_suffix=parsed[2])
        elif len(parsed) == 2:
            return cls('', deployment_name=parsed[0], replica_suffix=parsed[1])
        else:
            raise ValueError(f"Given replica tag {tag} didn't match pattern, please ensure it has either two or three fields with delimiter {cls.delimiter}")

    def __str__(self):
        return self.replica_tag