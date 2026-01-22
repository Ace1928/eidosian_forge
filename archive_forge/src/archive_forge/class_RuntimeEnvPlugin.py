import logging
import os
import json
from abc import ABC
from typing import List, Dict, Optional, Any, Type
from ray._private.runtime_env.context import RuntimeEnvContext
from ray._private.runtime_env.uri_cache import URICache
from ray._private.runtime_env.constants import (
from ray.util.annotations import DeveloperAPI
from ray._private.utils import import_attr
@DeveloperAPI
class RuntimeEnvPlugin(ABC):
    """Abstract base class for runtime environment plugins."""
    name: str = None
    priority: int = RAY_RUNTIME_ENV_PLUGIN_DEFAULT_PRIORITY

    @staticmethod
    def validate(runtime_env_dict: dict) -> None:
        """Validate user entry for this plugin.

        The method is invoked upon installation of runtime env.

        Args:
            runtime_env_dict: the user-supplied runtime environment dict.

        Raises:
            ValueError: if the validation fails.
        """
        pass

    def get_uris(self, runtime_env: 'RuntimeEnv') -> List[str]:
        return []

    async def create(self, uri: Optional[str], runtime_env: 'RuntimeEnv', context: RuntimeEnvContext, logger: logging.Logger) -> float:
        """Create and install the runtime environment.

        Gets called in the runtime env agent at install time. The URI can be
        used as a caching mechanism.

        Args:
            uri: A URI uniquely describing this resource.
            runtime_env: The RuntimeEnv object.
            context: auxiliary information supplied by Ray.
            logger: A logger to log messages during the context modification.

        Returns:
            the disk space taken up by this plugin installation for this
            environment. e.g. for working_dir, this downloads the files to the
            local node.
        """
        return 0

    def modify_context(self, uris: List[str], runtime_env: 'RuntimeEnv', context: RuntimeEnvContext, logger: logging.Logger) -> None:
        """Modify context to change worker startup behavior.

        For example, you can use this to preprend "cd <dir>" command to worker
        startup, or add new environment variables.

        Args:
            uris: The URIs used by this resource.
            runtime_env: The RuntimeEnv object.
            context: Auxiliary information supplied by Ray.
            logger: A logger to log messages during the context modification.
        """
        return

    def delete_uri(self, uri: str, logger: logging.Logger) -> float:
        """Delete the the runtime environment given uri.

        Args:
            uri: a URI uniquely describing this resource.

        Returns:
            the amount of space reclaimed by the deletion.
        """
        return 0