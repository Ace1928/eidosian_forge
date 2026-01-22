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