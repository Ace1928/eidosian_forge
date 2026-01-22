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
def _set_target_state(self, deployment_infos: Optional[Dict[str, DeploymentInfo]], code_version: str, target_config: Optional[ServeApplicationSchema], target_capacity: Optional[float]=None, target_capacity_direction: Optional[TargetCapacityDirection]=None, deleting: bool=False):
    """Set application target state.

        While waiting for build task to finish, this should be
            (None, False)
        When build task has finished and during normal operation, this should be
            (target_deployments, False)
        When a request to delete the application has been received, this should be
            ({}, True)
        """
    if deleting:
        self._update_status(ApplicationStatus.DELETING)
    else:
        self._update_status(ApplicationStatus.DEPLOYING)
    if deployment_infos is None:
        self._ingress_deployment_name = None
    else:
        for name, info in deployment_infos.items():
            if info.ingress:
                self._ingress_deployment_name = name
    target_state = ApplicationTargetState(deployment_infos, code_version, target_config, target_capacity, target_capacity_direction, deleting)
    self._save_checkpoint_func(writeahead_checkpoints={self._name: target_state})
    self._target_state = target_state