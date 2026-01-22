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
@PublicAPI(stability='stable')
class Application(DAGNodeBase):
    """One or more deployments bound with arguments that can be deployed together.

    Can be passed into another `Deployment.bind()` to compose multiple deployments in a
    single application, passed to `serve.run`, or deployed via a Serve config file.

    For example, to define an Application and run it in Python:

        .. code-block:: python

            from ray import serve
            from ray.serve import Application

            @serve.deployment
            class MyDeployment:
                pass

            app: Application = MyDeployment.bind(OtherDeployment.bind())
            serve.run(app)

    To run the same app using the command line interface (CLI):

        .. code-block:: bash

            serve run python_file:app

    To deploy the same app via a config file:

        .. code-block:: yaml

            applications:
                my_app:
                    import_path: python_file:app

    """

    def __init__(self, *, _internal_dag_node: Optional[Union[ClassNode, FunctionNode]]=None):
        if _internal_dag_node is None:
            raise RuntimeError('This class should not be constructed directly.')
        self._internal_dag_node = _internal_dag_node

    def _get_internal_dag_node(self) -> Union[ClassNode, FunctionNode]:
        if self._internal_dag_node is None:
            raise RuntimeError('Application object should not be constructed directly.')
        return self._internal_dag_node

    @classmethod
    def _from_internal_dag_node(cls, dag_node: Union[ClassNode, FunctionNode]):
        return cls(_internal_dag_node=dag_node)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._get_internal_dag_node(), name)