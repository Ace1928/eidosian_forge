import inspect
import logging
import warnings
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from ray.dag.class_node import ClassNode
from ray.dag.dag_node import DAGNodeBase
from ray.dag.function_node import FunctionNode
from ray.serve._private.config import DeploymentConfig, ReplicaConfig
from ray.serve._private.constants import SERVE_LOGGER_NAME
from ray.serve._private.utils import DEFAULT, Default
from ray.serve.config import AutoscalingConfig
from ray.serve.context import _get_global_client
from ray.serve.handle import RayServeHandle, RayServeSyncHandle
from ray.serve.schema import DeploymentSchema, LoggingConfig, RayActorOptionsSchema
from ray.util.annotations import Deprecated, PublicAPI
def _deploy(self, *init_args, _blocking=True, **init_kwargs):
    """Deploy or update this deployment.

        Args:
            init_args: args to pass to the class __init__
                method. Not valid if this deployment wraps a function.
            init_kwargs: kwargs to pass to the class __init__
                method. Not valid if this deployment wraps a function.
        """
    if len(init_args) == 0 and self._replica_config.init_args is not None:
        init_args = self._replica_config.init_args
    if len(init_kwargs) == 0 and self._replica_config.init_kwargs is not None:
        init_kwargs = self._replica_config.init_kwargs
    replica_config = ReplicaConfig.create(self._replica_config.deployment_def, init_args=init_args, init_kwargs=init_kwargs, ray_actor_options=self._replica_config.ray_actor_options, placement_group_bundles=self._replica_config.placement_group_bundles, placement_group_strategy=self._replica_config.placement_group_strategy, max_replicas_per_node=self._replica_config.max_replicas_per_node)
    return _get_global_client().deploy(self._name, replica_config=replica_config, deployment_config=self._deployment_config, version=self._version, route_prefix=self.route_prefix, url=self.url, _blocking=_blocking)