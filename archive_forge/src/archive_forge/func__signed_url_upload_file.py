import base64
import logging
import os
import posixpath
import uuid
import requests
import mlflow.tracking
from mlflow.azure.client import (
from mlflow.entities import FileInfo
from mlflow.environment_variables import (
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_artifacts_pb2 import (
from mlflow.protos.databricks_pb2 import (
from mlflow.protos.service_pb2 import GetRun, ListArtifacts, MlflowService
from mlflow.store.artifact.cloud_artifact_repo import (
from mlflow.utils import chunk_list
from mlflow.utils.databricks_utils import get_databricks_host_creds
from mlflow.utils.file_utils import (
from mlflow.utils.proto_json_utils import message_to_json
from mlflow.utils.request_utils import cloud_storage_http_request
from mlflow.utils.rest_utils import (
from mlflow.utils.uri import (
def _signed_url_upload_file(self, credentials, local_file):
    try:
        headers = self._extract_headers_from_credentials(credentials.headers)
        signed_write_uri = credentials.signed_uri
        if os.stat(local_file).st_size == 0:
            with cloud_storage_http_request('put', signed_write_uri, data='', headers=headers) as response:
                augmented_raise_for_status(response)
        else:
            with open(local_file, 'rb') as file:
                with cloud_storage_http_request('put', signed_write_uri, data=file, headers=headers) as response:
                    augmented_raise_for_status(response)
    except Exception as err:
        raise MlflowException(err)