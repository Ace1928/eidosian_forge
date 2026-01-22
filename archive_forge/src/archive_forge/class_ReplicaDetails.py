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
class ReplicaDetails(ServeActorDetails, frozen=True):
    """Detailed info about a single deployment replica."""
    replica_id: str = Field(description='Unique ID for the replica. By default, this will be "<deployment name>#<replica suffix>", where the replica suffix is a randomly generated unique string.')
    state: ReplicaState = Field(description='Current state of the replica.')
    pid: Optional[int] = Field(description='PID of the replica actor process.')
    start_time_s: float = Field(description='The time at which the replica actor was started. If the controller dies, this is the time at which the controller recovers and retrieves replica state from the running replica actor.')