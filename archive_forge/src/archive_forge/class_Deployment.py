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
class Deployment:
    """Class (or function) decorated with the `@serve.deployment` decorator.

    This is run on a number of replica actors. Requests to those replicas call
    this class.

    One or more deployments can be composed together into an `Application` which is
    then run via `serve.run` or a config file.

    Example:

    .. code-block:: python

        @serve.deployment
        class MyDeployment:
            def __init__(self, name: str):
                self._name = name

            def __call__(self, request):
                return "Hello world!"

            app = MyDeployment.bind()
            # Run via `serve.run` or the `serve run` CLI command.
            serve.run(app)

    """

    def __init__(self, name: str, deployment_config: DeploymentConfig, replica_config: ReplicaConfig, version: Optional[str]=None, route_prefix: Union[str, None, DEFAULT]=DEFAULT.VALUE, _internal=False) -> None:
        if not _internal:
            raise RuntimeError('The Deployment constructor should not be called directly. Use `@serve.deployment` instead.')
        if not isinstance(name, str):
            raise TypeError('name must be a string.')
        if not (version is None or isinstance(version, str)):
            raise TypeError('version must be a string.')
        if route_prefix is not DEFAULT.VALUE and route_prefix is not None:
            if not isinstance(route_prefix, str):
                raise TypeError('route_prefix must be a string.')
            if not route_prefix.startswith('/'):
                raise ValueError("route_prefix must start with '/'.")
            if route_prefix != '/' and route_prefix.endswith('/'):
                raise ValueError("route_prefix must not end with '/' unless it's the root.")
            if '{' in route_prefix or '}' in route_prefix:
                raise ValueError('route_prefix may not contain wildcards.')
        docs_path = None
        if inspect.isclass(replica_config.deployment_def) and hasattr(replica_config.deployment_def, '__module__') and (replica_config.deployment_def.__module__ == 'ray.serve.api') and hasattr(replica_config.deployment_def, '__fastapi_docs_path__'):
            docs_path = replica_config.deployment_def.__fastapi_docs_path__
        self._name = name
        self._version = version
        self._deployment_config = deployment_config
        self._replica_config = replica_config
        self._route_prefix = route_prefix
        self._docs_path = docs_path

    @property
    def name(self) -> str:
        """Unique name of this deployment."""
        return self._name

    @property
    def version(self) -> Optional[str]:
        return self._version

    @property
    def func_or_class(self) -> Union[Callable, str]:
        """Underlying class or function that this deployment wraps."""
        return self._replica_config.deployment_def

    @property
    def num_replicas(self) -> int:
        """Current target number of replicas."""
        return self._deployment_config.num_replicas

    @property
    def user_config(self) -> Any:
        """Current dynamic user-provided config options."""
        return self._deployment_config.user_config

    @property
    def max_concurrent_queries(self) -> int:
        """Current max outstanding queries from each handle."""
        return self._deployment_config.max_concurrent_queries

    @property
    def route_prefix(self) -> Optional[str]:
        """HTTP route prefix that this deployment is exposed under."""
        if self._route_prefix is DEFAULT.VALUE:
            return f'/{self._name}'
        return self._route_prefix

    @property
    def ray_actor_options(self) -> Optional[Dict]:
        """Actor options such as resources required for each replica."""
        return self._replica_config.ray_actor_options

    @property
    def init_args(self) -> Tuple[Any]:
        return self._replica_config.init_args

    @property
    def init_kwargs(self) -> Tuple[Any]:
        return self._replica_config.init_kwargs

    @property
    def url(self) -> Optional[str]:
        if self._route_prefix is None:
            return None
        return _get_global_client().root_url + self.route_prefix

    @property
    def logging_config(self) -> Dict:
        return self._deployment_config.logging_config

    def set_logging_config(self, logging_config: Dict):
        self._deployment_config.logging_config = logging_config

    def __call__(self):
        raise RuntimeError('Deployments cannot be constructed directly. Use `deployment.deploy() instead.`')

    def bind(self, *args, **kwargs) -> Application:
        """Bind the arguments to the deployment and return an Application.

        The returned Application can be deployed using `serve.run` (or via
        config file) or bound to another deployment for composition.
        """
        schema_shell = deployment_to_schema(self)
        if inspect.isfunction(self.func_or_class):
            dag_node = FunctionNode(self.func_or_class, args, kwargs, self._replica_config.ray_actor_options or dict(), other_args_to_resolve={'deployment_schema': schema_shell, 'is_from_serve_deployment': True})
        else:
            dag_node = ClassNode(self.func_or_class, args, kwargs, cls_options=self._replica_config.ray_actor_options or dict(), other_args_to_resolve={'deployment_schema': schema_shell, 'is_from_serve_deployment': True})
        return Application._from_internal_dag_node(dag_node)

    def deploy(self, *init_args, _blocking=True, **init_kwargs):
        raise ValueError('This API has been fully deprecated. Please use serve.run() instead.')

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

    def delete(self):
        raise ValueError('This API has been fully deprecated. Please use serve.run() and serve.delete() instead.')

    def _delete(self):
        """Delete this deployment."""
        return _get_global_client().delete_deployments([self._name])

    def get_handle(self, sync: Optional[bool]=True) -> Union[RayServeHandle, RayServeSyncHandle]:
        raise ValueError('This API has been fully deprecated. Please use serve.get_app_handle() or serve.get_deployment_handle() instead.')

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

    def options(self, func_or_class: Optional[Callable]=None, name: Default[str]=DEFAULT.VALUE, version: Default[str]=DEFAULT.VALUE, num_replicas: Default[Optional[int]]=DEFAULT.VALUE, route_prefix: Default[Union[str, None]]=DEFAULT.VALUE, ray_actor_options: Default[Optional[Dict]]=DEFAULT.VALUE, placement_group_bundles: Optional[List[Dict[str, float]]]=DEFAULT.VALUE, placement_group_strategy: Optional[str]=DEFAULT.VALUE, max_replicas_per_node: Optional[int]=DEFAULT.VALUE, user_config: Default[Optional[Any]]=DEFAULT.VALUE, max_concurrent_queries: Default[int]=DEFAULT.VALUE, autoscaling_config: Default[Union[Dict, AutoscalingConfig, None]]=DEFAULT.VALUE, graceful_shutdown_wait_loop_s: Default[float]=DEFAULT.VALUE, graceful_shutdown_timeout_s: Default[float]=DEFAULT.VALUE, health_check_period_s: Default[float]=DEFAULT.VALUE, health_check_timeout_s: Default[float]=DEFAULT.VALUE, logging_config: Default[Union[Dict, LoggingConfig, None]]=DEFAULT.VALUE, _init_args: Default[Tuple[Any]]=DEFAULT.VALUE, _init_kwargs: Default[Dict[Any, Any]]=DEFAULT.VALUE, _internal: bool=False) -> 'Deployment':
        """Return a copy of this deployment with updated options.

        Only those options passed in will be updated, all others will remain
        unchanged from the existing deployment.

        Refer to the `@serve.deployment` decorator docs for available arguments.
        """
        user_configured_option_names = [option for option, value in locals().items() if option not in {'self', 'func_or_class', '_internal'} and value is not DEFAULT.VALUE]
        new_deployment_config = deepcopy(self._deployment_config)
        if not _internal:
            new_deployment_config.user_configured_option_names.update(user_configured_option_names)
        if num_replicas not in [DEFAULT.VALUE, None] and autoscaling_config not in [DEFAULT.VALUE, None]:
            raise ValueError('Manually setting num_replicas is not allowed when autoscaling_config is provided.')
        if num_replicas == 0:
            raise ValueError('num_replicas is expected to larger than 0')
        if not _internal and version is not DEFAULT.VALUE:
            logger.warning('DeprecationWarning: `version` in `Deployment.options()` has been deprecated. Explicitly specifying version will raise an error in the future!')
        if not _internal and route_prefix is not DEFAULT.VALUE:
            logger.warning('DeprecationWarning: `route_prefix` in `@serve.deployment` has been deprecated. To specify a route prefix for an application, pass it into `serve.run` instead.')
        if num_replicas not in [DEFAULT.VALUE, None]:
            new_deployment_config.num_replicas = num_replicas
        if user_config is not DEFAULT.VALUE:
            new_deployment_config.user_config = user_config
        if max_concurrent_queries is not DEFAULT.VALUE:
            new_deployment_config.max_concurrent_queries = max_concurrent_queries
        if func_or_class is None:
            func_or_class = self._replica_config.deployment_def
        if name is DEFAULT.VALUE:
            name = self._name
        if version is DEFAULT.VALUE:
            version = self._version
        if _init_args is DEFAULT.VALUE:
            _init_args = self._replica_config.init_args
        if _init_kwargs is DEFAULT.VALUE:
            _init_kwargs = self._replica_config.init_kwargs
        if route_prefix is DEFAULT.VALUE:
            route_prefix = self._route_prefix
        if ray_actor_options is DEFAULT.VALUE:
            ray_actor_options = self._replica_config.ray_actor_options
        if placement_group_bundles is DEFAULT.VALUE:
            placement_group_bundles = self._replica_config.placement_group_bundles
        if placement_group_strategy is DEFAULT.VALUE:
            placement_group_strategy = self._replica_config.placement_group_strategy
        if max_replicas_per_node is DEFAULT.VALUE:
            max_replicas_per_node = self._replica_config.max_replicas_per_node
        if autoscaling_config is not DEFAULT.VALUE:
            new_deployment_config.autoscaling_config = autoscaling_config
        if graceful_shutdown_wait_loop_s is not DEFAULT.VALUE:
            new_deployment_config.graceful_shutdown_wait_loop_s = graceful_shutdown_wait_loop_s
        if graceful_shutdown_timeout_s is not DEFAULT.VALUE:
            new_deployment_config.graceful_shutdown_timeout_s = graceful_shutdown_timeout_s
        if health_check_period_s is not DEFAULT.VALUE:
            new_deployment_config.health_check_period_s = health_check_period_s
        if health_check_timeout_s is not DEFAULT.VALUE:
            new_deployment_config.health_check_timeout_s = health_check_timeout_s
        if logging_config is not DEFAULT.VALUE:
            if isinstance(logging_config, LoggingConfig):
                logging_config = logging_config.dict()
            new_deployment_config.logging_config = logging_config
        new_replica_config = ReplicaConfig.create(func_or_class, init_args=_init_args, init_kwargs=_init_kwargs, ray_actor_options=ray_actor_options, placement_group_bundles=placement_group_bundles, placement_group_strategy=placement_group_strategy, max_replicas_per_node=max_replicas_per_node)
        return Deployment(name, new_deployment_config, new_replica_config, version=version, route_prefix=route_prefix, _internal=True)

    @Deprecated(message='This was intended for use with the `serve.build` Python API (which has been deprecated). Use `.options()` instead.')
    def set_options(self, func_or_class: Optional[Callable]=None, name: Default[str]=DEFAULT.VALUE, version: Default[str]=DEFAULT.VALUE, num_replicas: Default[Optional[int]]=DEFAULT.VALUE, route_prefix: Default[Union[str, None]]=DEFAULT.VALUE, ray_actor_options: Default[Optional[Dict]]=DEFAULT.VALUE, user_config: Default[Optional[Any]]=DEFAULT.VALUE, max_concurrent_queries: Default[int]=DEFAULT.VALUE, autoscaling_config: Default[Union[Dict, AutoscalingConfig, None]]=DEFAULT.VALUE, graceful_shutdown_wait_loop_s: Default[float]=DEFAULT.VALUE, graceful_shutdown_timeout_s: Default[float]=DEFAULT.VALUE, health_check_period_s: Default[float]=DEFAULT.VALUE, health_check_timeout_s: Default[float]=DEFAULT.VALUE, _internal: bool=False) -> None:
        """Overwrite this deployment's options in-place.

        Only those options passed in will be updated, all others will remain
        unchanged.

        Refer to the @serve.deployment decorator docstring for all non-private
        arguments.
        """
        if not _internal:
            warnings.warn('`.set_options()` is deprecated. Use `.options()` or an application builder function instead.')
        validated = self.options(func_or_class=func_or_class, name=name, version=version, route_prefix=route_prefix, num_replicas=num_replicas, ray_actor_options=ray_actor_options, user_config=user_config, max_concurrent_queries=max_concurrent_queries, autoscaling_config=autoscaling_config, graceful_shutdown_wait_loop_s=graceful_shutdown_wait_loop_s, graceful_shutdown_timeout_s=graceful_shutdown_timeout_s, health_check_period_s=health_check_period_s, health_check_timeout_s=health_check_timeout_s, _internal=_internal)
        self._name = validated._name
        self._version = validated._version
        self._route_prefix = validated._route_prefix
        self._deployment_config = validated._deployment_config
        self._replica_config = validated._replica_config

    def __eq__(self, other):
        return all([self._name == other._name, self._version == other._version, self._deployment_config == other._deployment_config, self._replica_config.init_args == other._replica_config.init_args, self._replica_config.init_kwargs == other._replica_config.init_kwargs, self.route_prefix == other.route_prefix, self._replica_config.ray_actor_options == self._replica_config.ray_actor_options])

    def __str__(self):
        return f'Deployment(name={self._name},version={self._version},route_prefix={self.route_prefix})'

    def __repr__(self):
        return str(self)