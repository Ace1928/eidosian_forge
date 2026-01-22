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
def _reconcile_target_deployments(self) -> None:
    """Reconcile target deployments in application target state.

        Ensure each deployment is running on up-to-date info, and
        remove outdated deployments from the application.
        """
    for deployment_name, info in self._target_state.deployment_infos.items():
        deploy_info = deepcopy(info)
        deploy_info.set_target_capacity(new_target_capacity=self._target_state.target_capacity, new_target_capacity_direction=self._target_state.target_capacity_direction)
        if self._target_state.config and self._target_state.config.logging_config and (deploy_info.deployment_config.logging_config is None):
            deploy_info.deployment_config.logging_config = self._target_state.config.logging_config
        self.apply_deployment_info(deployment_name, deploy_info)
    for deployment_name in self._get_live_deployments():
        if deployment_name not in self.target_deployments:
            self._delete_deployment(deployment_name)