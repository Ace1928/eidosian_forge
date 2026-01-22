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
def _get_assumed_role_arn(**assume_role_credentials):
    """
    Returns:
        ARN of the user's current IAM role.
    """
    import boto3
    sess = boto3.Session()
    sts_client = sess.client('sts', **assume_role_credentials)
    identity_info = sts_client.get_caller_identity()
    sts_arn = identity_info['Arn']
    role_name = sts_arn.split('/')[1]
    iam_client = sess.client('iam', **assume_role_credentials)
    role_response = iam_client.get_role(RoleName=role_name)
    return role_response['Role']['Arn']