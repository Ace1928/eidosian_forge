import asyncio
import hashlib
import json
import logging
import os
import shutil
import sys
import tempfile
from typing import Dict, List, Optional, Tuple
from contextlib import asynccontextmanager
from asyncio import create_task, get_running_loop
from ray._private.runtime_env.context import RuntimeEnvContext
from ray._private.runtime_env.packaging import Protocol, parse_uri
from ray._private.runtime_env.plugin import RuntimeEnvPlugin
from ray._private.runtime_env.utils import check_output_cmd
from ray._private.utils import get_directory_size_bytes, try_to_create_directory
import ray
class PipPlugin(RuntimeEnvPlugin):
    name = 'pip'

    def __init__(self, resources_dir: str):
        self._pip_resources_dir = os.path.join(resources_dir, 'pip')
        self._creating_task = {}
        self._create_locks: Dict[str, asyncio.Lock] = {}
        self._created_hash_bytes: Dict[str, int] = {}
        try_to_create_directory(self._pip_resources_dir)

    def _get_path_from_hash(self, hash: str) -> str:
        """Generate a path from the hash of a pip spec.

        Example output:
            /tmp/ray/session_2021-11-03_16-33-59_356303_41018/runtime_resources
                /pip/ray-9a7972c3a75f55e976e620484f58410c920db091
        """
        return os.path.join(self._pip_resources_dir, hash)

    def get_uris(self, runtime_env: 'RuntimeEnv') -> List[str]:
        """Return the pip URI from the RuntimeEnv if it exists, else return []."""
        pip_uri = runtime_env.pip_uri()
        if pip_uri:
            return [pip_uri]
        return []

    def delete_uri(self, uri: str, logger: Optional[logging.Logger]=default_logger) -> int:
        """Delete URI and return the number of bytes deleted."""
        logger.info('Got request to delete pip URI %s', uri)
        protocol, hash = parse_uri(uri)
        if protocol != Protocol.PIP:
            raise ValueError(f'PipPlugin can only delete URIs with protocol pip. Received protocol {protocol}, URI {uri}')
        task = self._creating_task.pop(hash, None)
        if task is not None:
            task.cancel()
        del self._created_hash_bytes[hash]
        pip_env_path = self._get_path_from_hash(hash)
        local_dir_size = get_directory_size_bytes(pip_env_path)
        del self._create_locks[uri]
        try:
            shutil.rmtree(pip_env_path)
        except OSError as e:
            logger.warning(f'Error when deleting pip env {pip_env_path}: {str(e)}')
            return 0
        return local_dir_size

    async def create(self, uri: str, runtime_env: 'RuntimeEnv', context: RuntimeEnvContext, logger: Optional[logging.Logger]=default_logger) -> int:
        if not runtime_env.has_pip():
            return 0
        protocol, hash = parse_uri(uri)
        target_dir = self._get_path_from_hash(hash)

        async def _create_for_hash():
            await PipProcessor(target_dir, runtime_env, logger)
            loop = get_running_loop()
            return await loop.run_in_executor(None, get_directory_size_bytes, target_dir)
        if uri not in self._create_locks:
            self._create_locks[uri] = asyncio.Lock()
        async with self._create_locks[uri]:
            if hash in self._created_hash_bytes:
                return self._created_hash_bytes[hash]
            self._creating_task[hash] = task = create_task(_create_for_hash())
            task.add_done_callback(lambda _: self._creating_task.pop(hash, None))
            bytes = await task
            self._created_hash_bytes[hash] = bytes
            return bytes

    def modify_context(self, uris: List[str], runtime_env: 'RuntimeEnv', context: RuntimeEnvContext, logger: logging.Logger=default_logger):
        if not runtime_env.has_pip():
            return
        uri = uris[0]
        protocol, hash = parse_uri(uri)
        target_dir = self._get_path_from_hash(hash)
        virtualenv_python = _PathHelper.get_virtualenv_python(target_dir)
        if not os.path.exists(virtualenv_python):
            raise ValueError(f'Local directory {target_dir} for URI {uri} does not exist on the cluster. Something may have gone wrong while installing the runtime_env `pip` packages.')
        context.py_executable = virtualenv_python
        context.command_prefix += _PathHelper.get_virtualenv_activate_command(target_dir)