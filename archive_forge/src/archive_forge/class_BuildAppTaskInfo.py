import logging
import time
import traceback
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple
import ray
from ray import cloudpickle
from ray._private.utils import import_attr
from ray.exceptions import RuntimeEnvSetupError
from ray.serve._private.common import (
from ray.serve._private.config import DeploymentConfig
from ray.serve._private.constants import SERVE_LOGGER_NAME
from ray.serve._private.deploy_utils import (
from ray.serve._private.deployment_info import DeploymentInfo
from ray.serve._private.deployment_state import DeploymentStateManager
from ray.serve._private.endpoint_state import EndpointState
from ray.serve._private.storage.kv_store import KVStoreBase
from ray.serve._private.usage import ServeUsageTag
from ray.serve._private.utils import (
from ray.serve.exceptions import RayServeException
from ray.serve.generated.serve_pb2 import DeploymentLanguage
from ray.serve.schema import DeploymentDetails, ServeApplicationSchema
from ray.types import ObjectRef
@dataclass
class BuildAppTaskInfo:
    """Stores info on the current in-progress build app task.

    We use a class instead of only storing the task object ref because
    when a new config is deployed, there can be an outdated in-progress
    build app task. We attach the code version to the task info to
    distinguish outdated build app tasks.
    """
    obj_ref: ObjectRef
    code_version: str
    config: ServeApplicationSchema
    target_capacity: Optional[float]
    target_capacity_direction: Optional[TargetCapacityDirection]
    finished: bool