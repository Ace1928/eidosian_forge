import json
import logging
from mlflow.protos.databricks_pb2 import (
class _UnsupportedMultipartUploadException(MlflowException):
    """Exception thrown when multipart upload is unsupported by an artifact repository"""
    MESSAGE = 'Multipart upload is not supported for the current artifact repository'

    def __init__(self):
        super().__init__(self.MESSAGE, error_code=NOT_IMPLEMENTED)