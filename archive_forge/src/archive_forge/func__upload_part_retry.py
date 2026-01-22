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
def _upload_part_retry(self, cred_info, upload_id, part_number, local_file, start_byte, size):
    data = read_chunk(local_file, size, start_byte)
    try:
        return self._upload_part(cred_info, data)
    except requests.HTTPError as e:
        if e.response.status_code not in (401, 403):
            raise e
        _logger.info('Failed to authorize request, possibly due to credential expiration. Refreshing credentials and trying again...')
        resp = self._get_presigned_upload_part_url(cred_info.run_id, cred_info.path, upload_id, part_number)
        return self._upload_part(resp.upload_credential_info, data)