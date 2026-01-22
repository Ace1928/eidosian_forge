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
@dataclass(eq=True)
class StatusOverview:
    app_status: ApplicationStatusInfo
    name: str = ''
    deployment_statuses: List[DeploymentStatusInfo] = field(default_factory=list)

    def debug_string(self):
        return json.dumps(asdict(self), indent=4)

    def get_deployment_status(self, name: str) -> Optional[DeploymentStatusInfo]:
        """Get a deployment's status by name.

        Args:
            name: Deployment's name.

        Return (Optional[DeploymentStatusInfo]): Status with a name matching
            the argument, if one exists. Otherwise, returns None.
        """
        for deployment_status in self.deployment_statuses:
            if name == deployment_status.name:
                return deployment_status
        return None

    def to_proto(self):
        app_status_proto = self.app_status.to_proto()
        deployment_status_protos = map(lambda status: status.to_proto(), self.deployment_statuses)
        deployment_status_proto_list = DeploymentStatusInfoListProto()
        deployment_status_proto_list.deployment_status_infos.extend(deployment_status_protos)
        return StatusOverviewProto(name=self.name, app_status=app_status_proto, deployment_statuses=deployment_status_proto_list)

    @classmethod
    def from_proto(cls, proto: StatusOverviewProto) -> 'StatusOverview':
        app_status = ApplicationStatusInfo.from_proto(proto.app_status)
        deployment_statuses = []
        for info_proto in proto.deployment_statuses.deployment_status_infos:
            deployment_statuses.append(DeploymentStatusInfo.from_proto(info_proto))
        return cls(app_status=app_status, deployment_statuses=deployment_statuses, name=proto.name)