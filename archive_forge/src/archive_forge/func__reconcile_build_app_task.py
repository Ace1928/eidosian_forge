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
def _reconcile_build_app_task(self) -> Tuple[Tuple, BuildAppStatus, str]:
    """If necessary, reconcile the in-progress build task.

        Returns:
            Deploy arguments (Dict[str, DeploymentInfo]):
                The deploy arguments returned from the build app task
                and their code version.
            Status (BuildAppStatus):
                NO_TASK_IN_PROGRESS: There is no build task to reconcile.
                SUCCEEDED: Task finished successfully.
                FAILED: An error occurred during execution of build app task
                IN_PROGRESS: Task hasn't finished yet.
            Error message (str):
                Non-empty string if status is DEPLOY_FAILED or UNHEALTHY
        """
    if self._build_app_task_info is None or self._build_app_task_info.finished:
        return (None, BuildAppStatus.NO_TASK_IN_PROGRESS, '')
    if not check_obj_ref_ready_nowait(self._build_app_task_info.obj_ref):
        return (None, BuildAppStatus.IN_PROGRESS, '')
    self._build_app_task_info.finished = True
    try:
        args, err = ray.get(self._build_app_task_info.obj_ref)
        if err is None:
            logger.info(f"Built application '{self._name}' successfully.")
        else:
            return (None, BuildAppStatus.FAILED, f"Deploying app '{self._name}' failed with exception:\n{err}")
    except RuntimeEnvSetupError:
        error_msg = f"Runtime env setup for app '{self._name}' failed:\n" + traceback.format_exc()
        return (None, BuildAppStatus.FAILED, error_msg)
    except Exception:
        error_msg = f"Unexpected error occured while deploying application '{self._name}': \n{traceback.format_exc()}"
        return (None, BuildAppStatus.FAILED, error_msg)
    try:
        deployment_infos = {params['deployment_name']: deploy_args_to_deployment_info(**params, app_name=self._name) for params in args}
        overrided_infos = override_deployment_info(self._name, deployment_infos, self._build_app_task_info.config)
        self._route_prefix, self._docs_path = self._check_routes(overrided_infos)
        return (overrided_infos, BuildAppStatus.SUCCEEDED, '')
    except (TypeError, ValueError, RayServeException):
        return (None, BuildAppStatus.FAILED, traceback.format_exc())
    except Exception:
        error_msg = f"Unexpected error occured while applying config for application '{self._name}': \n{traceback.format_exc()}"
        return (None, BuildAppStatus.FAILED, error_msg)