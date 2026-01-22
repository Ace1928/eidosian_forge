import logging
import os
import sys
import time
from abc import ABCMeta, abstractmethod
from configparser import ConfigParser
from os.path import expanduser, join
@staticmethod
def _get_databricks_model_serving_config():
    from mlflow.utils.databricks_utils import get_model_dependency_oauth_token
    OAUTH_CACHE_REFRESH_DURATION_SEC = 5 * 60
    OAUTH_CACHE_ENV_VAR = 'DATABRICKS_DEPENDENCY_OAUTH_CACHE'
    OAUTH_CACHE_EXPIRATION_ENV_VAR = 'DATABRICKS_DEPENDENCY_OAUTH_CACHE_EXIRY_TS'
    MODEL_SERVING_HOST_ENV_VAR = 'DATABRICKS_MODEL_SERVING_HOST_URL'
    oauth_token = ''
    if OAUTH_CACHE_ENV_VAR in os.environ and OAUTH_CACHE_EXPIRATION_ENV_VAR in os.environ and (float(os.environ[OAUTH_CACHE_EXPIRATION_ENV_VAR]) > time.time()):
        oauth_token = os.environ[OAUTH_CACHE_ENV_VAR]
    else:
        oauth_token = get_model_dependency_oauth_token()
        os.environ[OAUTH_CACHE_ENV_VAR] = oauth_token
        os.environ[OAUTH_CACHE_EXPIRATION_ENV_VAR] = str(time.time() + OAUTH_CACHE_REFRESH_DURATION_SEC)
    return DatabricksConfig(host=os.environ[MODEL_SERVING_HOST_ENV_VAR], token=oauth_token, username=None, password=None, refresh_token=None, insecure=None, jobs_api_version=None)