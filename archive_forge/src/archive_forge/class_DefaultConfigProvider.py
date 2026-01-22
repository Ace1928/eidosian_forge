import logging
import os
import sys
import time
from abc import ABCMeta, abstractmethod
from configparser import ConfigParser
from os.path import expanduser, join
class DefaultConfigProvider(DatabricksConfigProvider):
    """Look for credentials in a chain of default locations."""

    def __init__(self):
        self._providers = (SparkTaskContextConfigProvider(), EnvironmentVariableConfigProvider(), ProfileConfigProvider(), DatabricksModelServingConfigProvider())

    def get_config(self):
        for provider in self._providers:
            config = provider.get_config()
            if config is not None and config.is_valid:
                return config
        return None