import json
import logging
from mlflow.protos.databricks_pb2 import (
class ExecutionException(MlflowException):
    """Exception thrown when executing a project fails"""
    pass