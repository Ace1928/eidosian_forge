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
def _azure_upload_file(self, credentials, local_file, artifact_file_path):
    """
        Uploads a file to a given Azure storage location.
        The function uses a file chunking generator with 100 MB being the size limit for each chunk.
        This limit is imposed by the stage_block API in azure-storage-blob.
        In the case the file size is large and the upload takes longer than the validity of the
        given credentials, a new set of credentials are generated and the operation continues. This
        is the reason for the first nested try-except block
        Finally, since the prevailing credentials could expire in the time between the last
        stage_block and the commit, a second try-except block refreshes credentials if needed.
        """
    try:
        headers = self._extract_headers_from_credentials(credentials.headers)
        futures = {}
        num_chunks = _compute_num_chunks(local_file, MLFLOW_MULTIPART_UPLOAD_CHUNK_SIZE.get())
        for index in range(num_chunks):
            start_byte = index * MLFLOW_MULTIPART_UPLOAD_CHUNK_SIZE.get()
            future = self.chunk_thread_pool.submit(self._azure_upload_chunk, credentials=credentials, headers=headers, local_file=local_file, artifact_file_path=artifact_file_path, start_byte=start_byte, size=MLFLOW_MULTIPART_UPLOAD_CHUNK_SIZE.get())
            futures[future] = index
        results, errors = _complete_futures(futures, local_file)
        if errors:
            raise MlflowException(f'Failed to upload at least one part of {local_file}. Errors: {errors}')
        uploading_block_list = [results[index] for index in sorted(results)]
        try:
            put_block_list(credentials.signed_uri, uploading_block_list, headers=headers)
        except requests.HTTPError as e:
            if e.response.status_code in [401, 403]:
                _logger.info('Failed to authorize request, possibly due to credential expiration. Refreshing credentials and trying again...')
                credential_info = self._get_write_credential_infos([artifact_file_path])[0]
                put_block_list(credential_info.signed_uri, uploading_block_list, headers=headers)
            else:
                raise e
    except Exception as err:
        raise MlflowException(err)