import json
import logging
import os
import platform
import signal
import sys
import tarfile
import time
import urllib.parse
from subprocess import Popen
from typing import Any, Dict, List, Optional
import mlflow
import mlflow.version
from mlflow import mleap, pyfunc
from mlflow.deployments import BaseDeploymentClient, PredictionsResponse
from mlflow.environment_variables import (
from mlflow.exceptions import MlflowException
from mlflow.models import Model
from mlflow.models.container import (
from mlflow.models.container import (
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE, RESOURCE_DOES_NOT_EXIST
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils import get_unique_resource_id
from mlflow.utils.file_utils import TempDir
from mlflow.utils.proto_json_utils import dump_input_data
def _get_values_from_target_uri(self):
    parsed = urllib.parse.urlparse(self.target_uri)
    values_str = parsed.path.strip('/')
    if not parsed.scheme or not values_str:
        return
    separator_index = values_str.find('/')
    if separator_index == -1:
        self.region_name = values_str
    else:
        self.region_name = values_str[:separator_index]
        self.assumed_role_arn = values_str[separator_index + 1:]
        self.assumed_role_arn = self.assumed_role_arn.strip('/')
    if self.region_name.startswith('arn'):
        raise MlflowException(message=f'It looks like the target_uri contains an IAM role ARN without a region name.\nA region name must be provided when the target_uri contains a role ARN.\nIn this case, the target_uri must follow the format: sagemaker:/region_name/assumed_role_arn.\nThe provided target_uri is: {self.target_uri}\n', error_code=INVALID_PARAMETER_VALUE)