import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from ray._private.runtime_env.context import RuntimeEnvContext
from ray._private.runtime_env.packaging import (
from ray._private.runtime_env.plugin import RuntimeEnvPlugin
from ray._private.utils import get_directory_size_bytes, try_to_create_directory
from ray.exceptions import RuntimeEnvSetupError
class WorkingDirPlugin(RuntimeEnvPlugin):
    name = 'working_dir'

    def __init__(self, resources_dir: str, gcs_aio_client: 'GcsAioClient'):
        self._resources_dir = os.path.join(resources_dir, 'working_dir_files')
        self._gcs_aio_client = gcs_aio_client
        try_to_create_directory(self._resources_dir)

    def delete_uri(self, uri: str, logger: Optional[logging.Logger]=default_logger) -> int:
        """Delete URI and return the number of bytes deleted."""
        logger.info('Got request to delete working dir URI %s', uri)
        local_dir = get_local_dir_from_uri(uri, self._resources_dir)
        local_dir_size = get_directory_size_bytes(local_dir)
        deleted = delete_package(uri, self._resources_dir)
        if not deleted:
            logger.warning(f'Tried to delete nonexistent URI: {uri}.')
            return 0
        return local_dir_size

    def get_uris(self, runtime_env: 'RuntimeEnv') -> List[str]:
        working_dir_uri = runtime_env.working_dir()
        if working_dir_uri != '':
            return [working_dir_uri]
        return []

    async def create(self, uri: Optional[str], runtime_env: dict, context: RuntimeEnvContext, logger: logging.Logger=default_logger) -> int:
        local_dir = await download_and_unpack_package(uri, self._resources_dir, self._gcs_aio_client, logger=logger)
        return get_directory_size_bytes(local_dir)

    def modify_context(self, uris: List[str], runtime_env_dict: Dict, context: RuntimeEnvContext, logger: Optional[logging.Logger]=default_logger):
        if not uris:
            return
        uri = uris[0]
        local_dir = get_local_dir_from_uri(uri, self._resources_dir)
        if not local_dir.exists():
            raise ValueError(f'Local directory {local_dir} for URI {uri} does not exist on the cluster. Something may have gone wrong while downloading or unpacking the working_dir.')
        if not _WIN32:
            context.command_prefix += ['cd', str(local_dir), '&&']
        else:
            context.command_prefix += ['cd', '/d', f'{local_dir}', '&&']
        set_pythonpath_in_context(python_path=str(local_dir), context=context)