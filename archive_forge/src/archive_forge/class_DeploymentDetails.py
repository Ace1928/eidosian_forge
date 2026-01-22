import logging
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union
from zlib import crc32
from ray._private.pydantic_compat import (
from ray._private.runtime_env.packaging import parse_uri
from ray.serve._private.common import (
from ray.serve._private.constants import (
from ray.serve._private.deployment_info import DeploymentInfo
from ray.serve._private.utils import DEFAULT
from ray.serve.config import ProxyLocation
from ray.util.annotations import PublicAPI
@PublicAPI(stability='stable')
class DeploymentDetails(BaseModel, extra=Extra.forbid, frozen=True):
    """
    Detailed info about a deployment within a Serve application.
    """
    name: str = Field(description='Deployment name.')
    status: DeploymentStatus = Field(description='The current status of the deployment.')
    status_trigger: DeploymentStatusTrigger = Field(description='[EXPERIMENTAL] The trigger for the current status.')
    message: str = Field(description='If there are issues with the deployment, this will describe the issue in more detail.')
    deployment_config: DeploymentSchema = Field(description="The set of deployment config options that are currently applied to this deployment. These options may come from the user's code, config file options, or Serve default values.")
    replicas: List[ReplicaDetails] = Field(description='Details about the live replicas of this deployment.')

    @validator('deployment_config')
    def deployment_route_prefix_not_set(cls, v: DeploymentSchema):
        if 'route_prefix' in v.dict(exclude_unset=True):
            raise ValueError(f'Unexpectedly found a deployment-level route_prefix in the deployment_config for deployment "{cls.name}". The route_prefix in deployment_config within DeploymentDetails should not be set; please set it at the application level.')
        return v