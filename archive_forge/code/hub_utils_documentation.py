import functools
import logging
import os
from typing import Optional
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST

    Check if the given string is a valid HuggingFace repo identifier e.g. "username/repo_id".
    