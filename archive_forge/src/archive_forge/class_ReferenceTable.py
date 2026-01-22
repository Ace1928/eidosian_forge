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
class ReferenceTable:
    """
    The URI reference table which is used for GC.
    When the reference count is decreased to zero,
    the URI should be removed from this table and
    added to cache if needed.
    """

    def __init__(self, uris_parser: Callable[[RuntimeEnv], Tuple[str, UriType]], unused_uris_callback: Callable[[List[Tuple[str, UriType]]], None], unused_runtime_env_callback: Callable[[str], None]):
        self._runtime_env_reference: Dict[str, int] = defaultdict(int)
        self._uri_reference: Dict[str, int] = defaultdict(int)
        self._uris_parser = uris_parser
        self._unused_uris_callback = unused_uris_callback
        self._unused_runtime_env_callback = unused_runtime_env_callback
        self._reference_exclude_sources: Set[str] = {'client_server'}

    def _increase_reference_for_uris(self, uris):
        default_logger.debug(f'Increase reference for uris {uris}.')
        for uri, _ in uris:
            self._uri_reference[uri] += 1

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

    def _increase_reference_for_runtime_env(self, serialized_env: str):
        default_logger.debug(f'Increase reference for runtime env {serialized_env}.')
        self._runtime_env_reference[serialized_env] += 1

    def _decrease_reference_for_runtime_env(self, serialized_env: str):
        default_logger.debug(f'Decrease reference for runtime env {serialized_env}.')
        unused = False
        if self._runtime_env_reference[serialized_env] > 0:
            self._runtime_env_reference[serialized_env] -= 1
            if self._runtime_env_reference[serialized_env] == 0:
                unused = True
                del self._runtime_env_reference[serialized_env]
        else:
            default_logger.warn(f'Runtime env {serialized_env} does not exist.')
        if unused:
            default_logger.info(f'Unused runtime env {serialized_env}.')
            self._unused_runtime_env_callback(serialized_env)
        return unused

    def increase_reference(self, runtime_env: RuntimeEnv, serialized_env: str, source_process: str) -> None:
        if source_process in self._reference_exclude_sources:
            return
        self._increase_reference_for_runtime_env(serialized_env)
        uris = self._uris_parser(runtime_env)
        self._increase_reference_for_uris(uris)

    def decrease_reference(self, runtime_env: RuntimeEnv, serialized_env: str, source_process: str) -> None:
        if source_process in self._reference_exclude_sources:
            return list()
        self._decrease_reference_for_runtime_env(serialized_env)
        uris = self._uris_parser(runtime_env)
        self._decrease_reference_for_uris(uris)

    @property
    def runtime_env_refs(self) -> Dict[str, int]:
        """Return the runtime_env -> ref count mapping.

        Returns:
            The mapping of serialized runtime env -> ref count.
        """
        return self._runtime_env_reference