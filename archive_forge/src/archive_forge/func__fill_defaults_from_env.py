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
def _fill_defaults_from_env(self):
    namespace_env_var = os.environ.get(RAY_NAMESPACE_ENVIRONMENT_VARIABLE)
    if namespace_env_var and self._job_config.ray_namespace is None:
        self.namespace(namespace_env_var)
    runtime_env_var = os.environ.get(RAY_RUNTIME_ENV_ENVIRONMENT_VARIABLE)
    if runtime_env_var and self._job_config.runtime_env is None:
        self.env(json.loads(runtime_env_var))