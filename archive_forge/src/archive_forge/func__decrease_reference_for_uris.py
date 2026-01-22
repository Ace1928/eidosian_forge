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
def _decrease_reference_for_uris(self, uris):
    default_logger.debug(f'Decrease reference for uris {uris}.')
    unused_uris = list()
    for uri, uri_type in uris:
        if self._uri_reference[uri] > 0:
            self._uri_reference[uri] -= 1
            if self._uri_reference[uri] == 0:
                unused_uris.append((uri, uri_type))
                del self._uri_reference[uri]
        else:
            default_logger.warn(f'URI {uri} does not exist.')
    if unused_uris:
        default_logger.info(f'Unused uris {unused_uris}.')
        self._unused_uris_callback(unused_uris)
    return unused_uris