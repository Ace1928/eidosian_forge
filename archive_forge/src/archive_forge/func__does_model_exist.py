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
def _does_model_exist(model_name, sage_client):
    """
    Determines whether a SageMaker model exists with the specified name in the caller's AWS account,
    returning True if the model exists, returning False if the model does not exist.

    Args:
        sage_client: A boto3 client for SageMaker.

    Returns:
        If the model exists, ``True``. If the model does not
        exist, ``False``.
    """
    try:
        response = sage_client.describe_model(ModelName=model_name)
    except sage_client.exceptions.ClientError as error:
        if 'Could not find model' in error.response['Error']['Message']:
            return False
    else:
        return bool(response)