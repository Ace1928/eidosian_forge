import logging
import os
import sys
import time
from abc import ABCMeta, abstractmethod
from configparser import ConfigParser
from os.path import expanduser, join
class EnvironmentVariableConfigProvider(DatabricksConfigProvider):
    """Loads from system environment variables."""

    def get_config(self):
        host = os.environ.get('DATABRICKS_HOST')
        username = os.environ.get('DATABRICKS_USERNAME')
        password = os.environ.get('DATABRICKS_PASSWORD')
        token = os.environ.get('DATABRICKS_TOKEN')
        refresh_token = os.environ.get('DATABRICKS_REFRESH_TOKEN')
        insecure = os.environ.get('DATABRICKS_INSECURE')
        jobs_api_version = os.environ.get('DATABRICKS_JOBS_API_VERSION')
        config = DatabricksConfig(host, username, password, token, refresh_token, insecure, jobs_api_version)
        if config.is_valid:
            return config
        return None