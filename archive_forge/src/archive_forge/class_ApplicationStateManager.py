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
class ApplicationStateManager:

    def __init__(self, deployment_state_manager: DeploymentStateManager, endpoint_state: EndpointState, kv_store: KVStoreBase):
        self._deployment_state_manager = deployment_state_manager
        self._endpoint_state = endpoint_state
        self._kv_store = kv_store
        self._application_states: Dict[str, ApplicationState] = dict()
        self._recover_from_checkpoint()

    def _recover_from_checkpoint(self):
        checkpoint = self._kv_store.get(CHECKPOINT_KEY)
        if checkpoint is not None:
            application_state_info = cloudpickle.loads(checkpoint)
            for app_name, checkpoint_data in application_state_info.items():
                app_state = ApplicationState(app_name, self._deployment_state_manager, self._endpoint_state, self._save_checkpoint_func)
                app_state.recover_target_state_from_checkpoint(checkpoint_data)
                self._application_states[app_name] = app_state

    def delete_application(self, name: str) -> None:
        """Delete application by name"""
        if name not in self._application_states:
            return
        self._application_states[name].delete()

    def apply_deployment_args(self, name: str, deployment_args: List[Dict]) -> None:
        """Apply list of deployment arguments to application target state.

        This function should only be called if the app is being deployed
        through serve.run instead of from a config.

        Args:
            name: application name
            deployment_args_list: arguments for deploying a list of deployments.

        Raises:
            RayServeException: If the list of deployments is trying to
                use a route prefix that is already used by another application
        """
        live_route_prefixes: Dict[str, str] = {self._application_states[app_name].route_prefix: app_name for app_name, app_state in self._application_states.items() if app_state.route_prefix is not None and (not app_state.status == ApplicationStatus.DELETING) and (name != app_name)}
        for deploy_param in deployment_args:
            deploy_app_prefix = deploy_param.get('route_prefix')
            if deploy_app_prefix in live_route_prefixes:
                raise RayServeException(f'Prefix {deploy_app_prefix} is being used by application "{live_route_prefixes[deploy_app_prefix]}". Failed to deploy application "{name}".')
        if name not in self._application_states:
            self._application_states[name] = ApplicationState(name, self._deployment_state_manager, self._endpoint_state, self._save_checkpoint_func)
        ServeUsageTag.NUM_APPS.record(str(len(self._application_states)))
        deployment_infos = {params['deployment_name']: deploy_args_to_deployment_info(**params, app_name=name) for params in deployment_args}
        self._application_states[name].deploy(deployment_infos)

    def deploy_config(self, name: str, app_config: ServeApplicationSchema, deployment_time: float=0, target_capacity: Optional[float]=None, target_capacity_direction: Optional[TargetCapacityDirection]=None) -> None:
        """Deploy application from config."""
        if name not in self._application_states:
            self._application_states[name] = ApplicationState(name, self._deployment_state_manager, endpoint_state=self._endpoint_state, save_checkpoint_func=self._save_checkpoint_func)
        ServeUsageTag.NUM_APPS.record(str(len(self._application_states)))
        self._application_states[name].deploy_config(app_config, target_capacity, target_capacity_direction, deployment_time=deployment_time)

    def get_deployments(self, app_name: str) -> List[str]:
        """Return all deployment names by app name"""
        if app_name not in self._application_states:
            return []
        return self._application_states[app_name].target_deployments

    def get_deployments_statuses(self, app_name: str) -> List[DeploymentStatusInfo]:
        """Return all deployment statuses by app name"""
        if app_name not in self._application_states:
            return []
        return self._application_states[app_name].get_deployments_statuses()

    def get_app_status(self, name: str) -> ApplicationStatus:
        if name not in self._application_states:
            return ApplicationStatus.NOT_STARTED
        return self._application_states[name].status

    def get_app_status_info(self, name: str) -> ApplicationStatusInfo:
        if name not in self._application_states:
            return ApplicationStatusInfo(ApplicationStatus.NOT_STARTED, message=f"Application {name} doesn't exist", deployment_timestamp=0)
        return self._application_states[name].get_application_status_info()

    def get_docs_path(self, app_name: str) -> Optional[str]:
        return self._application_states[app_name].docs_path

    def get_route_prefix(self, name: str) -> Optional[str]:
        return self._application_states[name].route_prefix

    def get_ingress_deployment_name(self, name: str) -> Optional[str]:
        if name not in self._application_states:
            return None
        return self._application_states[name].ingress_deployment

    def list_app_statuses(self) -> Dict[str, ApplicationStatusInfo]:
        """Return a dictionary with {app name: application info}"""
        return {name: self._application_states[name].get_application_status_info() for name in self._application_states}

    def list_deployment_details(self, name: str) -> Dict[str, DeploymentDetails]:
        """Gets detailed info on all deployments in specified application."""
        if name not in self._application_states:
            return {}
        return self._application_states[name].list_deployment_details()

    def update(self):
        """Update each application state"""
        apps_to_be_deleted = []
        for name, app in self._application_states.items():
            ready_to_be_deleted = app.update()
            if ready_to_be_deleted:
                apps_to_be_deleted.append(name)
                logger.debug(f"Application '{name}' deleted successfully.")
        if len(apps_to_be_deleted) > 0:
            for app_name in apps_to_be_deleted:
                del self._application_states[app_name]
            ServeUsageTag.NUM_APPS.record(str(len(self._application_states)))

    def shutdown(self) -> None:
        for app_state in self._application_states.values():
            app_state.delete()
        self._kv_store.delete(CHECKPOINT_KEY)

    def is_ready_for_shutdown(self) -> bool:
        """Return whether all applications have shut down.

        Iterate through all application states and check if all their applications
        are deleted.
        """
        return all((app_state.is_deleted() for app_state in self._application_states.values()))

    def _save_checkpoint_func(self, *, writeahead_checkpoints: Optional[Dict[str, ApplicationTargetState]]) -> None:
        """Write a checkpoint of all application states."""
        application_state_info = {app_name: app_state.get_checkpoint_data() for app_name, app_state in self._application_states.items()}
        if writeahead_checkpoints is not None:
            application_state_info.update(writeahead_checkpoints)
        self._kv_store.put(CHECKPOINT_KEY, cloudpickle.dumps(application_state_info))