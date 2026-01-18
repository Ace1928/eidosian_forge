import asyncio
import json
import logging
import os
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, List, Set, Tuple
from ray._private.ray_constants import (
import ray._private.runtime_env.agent.runtime_env_consts as runtime_env_consts
from ray._private.ray_logging import setup_component_logger
from ray._private.runtime_env.conda import CondaPlugin
from ray._private.runtime_env.container import ContainerManager
from ray._private.runtime_env.context import RuntimeEnvContext
from ray._private.runtime_env.java_jars import JavaJarsPlugin
from ray._private.runtime_env.pip import PipPlugin
from ray._private.gcs_utils import GcsAioClient
from ray._private.runtime_env.plugin import (
from ray._private.utils import get_or_create_event_loop
from ray._private.runtime_env.plugin import RuntimeEnvPluginManager
from ray._private.runtime_env.py_modules import PyModulesPlugin
from ray._private.runtime_env.working_dir import WorkingDirPlugin
from ray._private.runtime_env.nsight import NsightPlugin
from ray._private.runtime_env.mpi import MPIPlugin
from ray.core.generated import (
from ray.core.generated.runtime_env_common_pb2 import (
from ray.runtime_env import RuntimeEnv, RuntimeEnvConfig
def unused_runtime_env_processor(self, unused_runtime_env: str) -> None:

    def delete_runtime_env():
        del self._env_cache[unused_runtime_env]
        self._logger.info('Runtime env %s removed from env-level cache.', unused_runtime_env)
    if unused_runtime_env in self._env_cache:
        if not self._env_cache[unused_runtime_env].success:
            loop = get_or_create_event_loop()
            loop.call_later(runtime_env_consts.BAD_RUNTIME_ENV_CACHE_TTL_SECONDS, delete_runtime_env)
        else:
            delete_runtime_env()