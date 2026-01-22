import importlib
import inspect
import json
import logging
import os
import sys
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import ray.util.client_connect
from ray._private.ray_constants import (
from ray._private.utils import check_ray_client_dependencies_installed, split_address
from ray._private.worker import BaseContext
from ray._private.worker import init as ray_driver_init
from ray.job_config import JobConfig
from ray.util.annotations import Deprecated, PublicAPI
@Deprecated
class ClientBuilder:
    """
    Builder for a Ray Client connection. This class can be subclassed by
    custom builder classes to modify connection behavior to include additional
    features or altered semantics. One example is the ``_LocalClientBuilder``.
    """

    def __init__(self, address: Optional[str]) -> None:
        if not check_ray_client_dependencies_installed():
            raise ValueError('Ray Client requires pip package `ray[client]`. If you installed the minimal Ray (e.g. `pip install ray`), please reinstall by executing `pip install ray[client]`.')
        self.address = address
        self._job_config = JobConfig()
        self._remote_init_kwargs = {}
        self._allow_multiple_connections = False
        self._credentials = None
        self._metadata = None
        self._deprecation_warn_enabled = True

    def env(self, env: Dict[str, Any]) -> 'ClientBuilder':
        """
        Set an environment for the session.
        Args:
            env (Dict[st, Any]): A runtime environment to use for this
            connection. See :ref:`runtime-environments` for what values are
            accepted in this dict.
        """
        self._job_config.set_runtime_env(env)
        return self

    def namespace(self, namespace: str) -> 'ClientBuilder':
        """
        Sets the namespace for the session.
        Args:
            namespace: Namespace to use.
        """
        self._job_config.set_ray_namespace(namespace)
        return self

    def connect(self) -> ClientContext:
        """
        Begin a connection to the address passed in via ray.client(...).

        Returns:
            ClientInfo: Dataclass with information about the setting. This
                includes the server's version of Python & Ray as well as the
                dashboard_url.
        """
        if self._deprecation_warn_enabled:
            self._client_deprecation_warn()
        self._fill_defaults_from_env()
        default_cli_connected = ray.util.client.ray.is_connected()
        has_cli_connected = ray.util.client.num_connected_contexts() > 0
        if not self._allow_multiple_connections and (not default_cli_connected) and has_cli_connected:
            raise ValueError('The client has already connected to the cluster with allow_multiple=True. Please set allow_multiple=True to proceed')
        old_ray_cxt = None
        if self._allow_multiple_connections:
            old_ray_cxt = ray.util.client.ray.set_context(None)
        client_info_dict = ray.util.client_connect.connect(self.address, job_config=self._job_config, _credentials=self._credentials, ray_init_kwargs=self._remote_init_kwargs, metadata=self._metadata)
        dashboard_url = ray.util.client.ray._get_dashboard_url()
        cxt = ClientContext(dashboard_url=dashboard_url, python_version=client_info_dict['python_version'], ray_version=client_info_dict['ray_version'], ray_commit=client_info_dict['ray_commit'], protocol_version=client_info_dict['protocol_version'], _num_clients=client_info_dict['num_clients'], _context_to_restore=ray.util.client.ray.get_context())
        if self._allow_multiple_connections:
            ray.util.client.ray.set_context(old_ray_cxt)
        return cxt

    def _fill_defaults_from_env(self):
        namespace_env_var = os.environ.get(RAY_NAMESPACE_ENVIRONMENT_VARIABLE)
        if namespace_env_var and self._job_config.ray_namespace is None:
            self.namespace(namespace_env_var)
        runtime_env_var = os.environ.get(RAY_RUNTIME_ENV_ENVIRONMENT_VARIABLE)
        if runtime_env_var and self._job_config.runtime_env is None:
            self.env(json.loads(runtime_env_var))

    def _init_args(self, **kwargs) -> 'ClientBuilder':
        """
        When a client builder is constructed through ray.init, for example
        `ray.init(ray://..., namespace=...)`, all of the
        arguments passed into ray.init with non-default values are passed
        again into this method. Custom client builders can override this method
        to do their own handling/validation of arguments.
        """
        if kwargs.get('namespace') is not None:
            self.namespace(kwargs['namespace'])
            del kwargs['namespace']
        if kwargs.get('runtime_env') is not None:
            self.env(kwargs['runtime_env'])
            del kwargs['runtime_env']
        if kwargs.get('allow_multiple') is True:
            self._allow_multiple_connections = True
            del kwargs['allow_multiple']
        if '_credentials' in kwargs.keys():
            self._credentials = kwargs['_credentials']
            del kwargs['_credentials']
        if '_metadata' in kwargs.keys():
            self._metadata = kwargs['_metadata']
            del kwargs['_metadata']
        if kwargs:
            expected_sig = inspect.signature(ray_driver_init)
            extra_args = set(kwargs.keys()).difference(expected_sig.parameters.keys())
            if len(extra_args) > 0:
                raise RuntimeError('Got unexpected kwargs: {}'.format(', '.join(extra_args)))
            self._remote_init_kwargs = kwargs
            unknown = ', '.join(kwargs)
            logger.info(f'Passing the following kwargs to ray.init() on the server: {unknown}')
        return self

    def _client_deprecation_warn(self) -> None:
        """
        Generates a warning for user's if this ClientBuilder instance was
        created directly or through ray.client, instead of relying on
        internal methods (ray.init, or auto init)
        """
        namespace = self._job_config.ray_namespace
        runtime_env = self._job_config.runtime_env
        replacement_args = []
        if self.address:
            if isinstance(self, _LocalClientBuilder):
                replacement_args.append(f'"{self.address}"')
            else:
                replacement_args.append(f'"ray://{self.address}"')
        if namespace:
            replacement_args.append(f'namespace="{namespace}"')
        if runtime_env:
            replacement_args.append('runtime_env=<your_runtime_env>')
        args_str = ', '.join(replacement_args)
        replacement_call = f'ray.init({args_str})'
        warnings.warn(f'Starting a connection through `ray.client` will be deprecated in future ray versions in favor of `ray.init`. See the docs for more details: {CLIENT_DOCS_URL}. You can replace your call to `ray.client().connect()` with the following:\n      {replacement_call}\n', DeprecationWarning, stacklevel=3)