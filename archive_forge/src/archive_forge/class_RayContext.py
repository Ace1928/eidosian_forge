import atexit
import faulthandler
import functools
import inspect
import io
import json
import logging
import os
import sys
import threading
import time
import traceback
import urllib
import warnings
from abc import ABCMeta, abstractmethod
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import (
from urllib.parse import urlparse
import colorama
import setproctitle
from typing import Literal, Protocol
import ray
import ray._private.node
import ray._private.parameter
import ray._private.profiling as profiling
import ray._private.ray_constants as ray_constants
import ray._private.serialization as serialization
import ray._private.services as services
import ray._private.state
import ray._private.storage as storage
import ray.actor
import ray.cloudpickle as pickle  # noqa
import ray.job_config
import ray.remote_function
from ray import ActorID, JobID, Language, ObjectRef
from ray._raylet import raise_sys_exit_with_custom_error_message
from ray._raylet import ObjectRefGenerator, TaskID
from ray.runtime_env.runtime_env import _merge_runtime_env
from ray._private import ray_option_utils
from ray._private.client_mode_hook import client_mode_hook
from ray._private.function_manager import FunctionActorManager
from ray._private.inspect_util import is_cython
from ray._private.ray_logging import (
from ray._private.runtime_env.constants import RAY_JOB_CONFIG_JSON_ENV_VAR
from ray._private.runtime_env.py_modules import upload_py_modules_if_needed
from ray._private.runtime_env.working_dir import upload_working_dir_if_needed
from ray._private.runtime_env.setup_hook import (
from ray._private.storage import _load_class
from ray._private.utils import get_ray_doc_version
from ray.exceptions import ObjectStoreFullError, RayError, RaySystemError, RayTaskError
from ray.experimental.internal_kv import (
from ray.experimental import tqdm_ray
from ray.experimental.tqdm_ray import RAY_TQDM_MAGIC
from ray.util.annotations import Deprecated, DeveloperAPI, PublicAPI
from ray.util.debug import log_once
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from ray.util.tracing.tracing_helper import _import_from_string
from ray.widgets import Template
from ray.widgets.util import repr_with_fallback
@dataclass
class RayContext(BaseContext, Mapping):
    """
    Context manager for attached drivers.
    """
    dashboard_url: Optional[str]
    python_version: str
    ray_version: str
    ray_commit: str
    protocol_version: Optional[str]

    def __init__(self, address_info: Dict[str, Optional[str]]):
        super().__init__()
        self.dashboard_url = get_dashboard_url()
        self.python_version = '{}.{}.{}'.format(*sys.version_info[:3])
        self.ray_version = ray.__version__
        self.ray_commit = ray.__commit__
        self.protocol_version = None
        self.address_info = address_info

    def __getitem__(self, key):
        if log_once('ray_context_getitem'):
            warnings.warn(f'Accessing values through ctx["{key}"] is deprecated. Use ctx.address_info["{key}"] instead.', DeprecationWarning, stacklevel=2)
        return self.address_info[key]

    def __len__(self):
        if log_once('ray_context_len'):
            warnings.warn('len(ctx) is deprecated. Use len(ctx.address_info) instead.')
        return len(self.address_info)

    def __iter__(self):
        if log_once('ray_context_len'):
            warnings.warn('iter(ctx) is deprecated. Use iter(ctx.address_info) instead.')
        return iter(self.address_info)

    def __enter__(self) -> 'RayContext':
        return self

    def __exit__(self, *exc):
        ray.shutdown()

    def disconnect(self):
        ray.shutdown()