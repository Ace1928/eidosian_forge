import configparser
import getpass
import logging
import os
from typing import NamedTuple, Optional, Tuple
from mlflow.environment_variables import (
from mlflow.exceptions import MlflowException
from mlflow.utils.rest_utils import MlflowHostCreds
def _read_mlflow_creds_from_env() -> Tuple[Optional[str], Optional[str]]:
    return (MLFLOW_TRACKING_USERNAME.get(), MLFLOW_TRACKING_PASSWORD.get())