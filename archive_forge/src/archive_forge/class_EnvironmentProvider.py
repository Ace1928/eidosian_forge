import copy
import logging
import os
from botocore import utils
from botocore.exceptions import InvalidConfigError
class EnvironmentProvider(BaseProvider):
    """This class loads config values from environment variables."""

    def __init__(self, name, env):
        """Initialize with the keys in the dictionary to check.

        :type name: str
        :param name: The key with that name will be loaded and returned.

        :type env: dict
        :param env: Environment variables dictionary to get variables from.
        """
        self._name = name
        self._env = env

    def __deepcopy__(self, memo):
        return EnvironmentProvider(copy.deepcopy(self._name, memo), copy.deepcopy(self._env, memo))

    def provide(self):
        """Provide a config value from a source dictionary."""
        if self._name in self._env:
            return self._env[self._name]
        return None

    def __repr__(self):
        return f'EnvironmentProvider(name={self._name}, env={self._env})'