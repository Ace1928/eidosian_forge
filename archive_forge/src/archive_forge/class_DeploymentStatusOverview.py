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
@PublicAPI(stability='alpha')
@dataclass
class DeploymentStatusOverview:
    """Describes the status of a deployment.

    Attributes:
        status: The current status of the deployment.
        replica_states: A map indicating how many replicas there are of
            each replica state.
        message: A message describing the deployment status in more
            detail.
    """
    status: DeploymentStatus
    status_trigger: DeploymentStatusTrigger
    replica_states: Dict[ReplicaState, int]
    message: str