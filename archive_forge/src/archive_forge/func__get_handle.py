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
def _get_handle(self, sync: Optional[bool]=True) -> Union[RayServeHandle, RayServeSyncHandle]:
    """Get a ServeHandle to this deployment to invoke it from Python.

        Args:
            sync: If true, then Serve will return a ServeHandle that
                works everywhere. Otherwise, Serve will return an
                asyncio-optimized ServeHandle that's only usable in an asyncio
                loop.

        Returns:
            ServeHandle
        """
    return _get_global_client().get_handle(self._name, app_name='', missing_ok=True, sync=sync, use_new_handle_api=False)