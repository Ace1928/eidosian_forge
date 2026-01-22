import logging
import os
import posixpath
import requests
from requests import HTTPError
from mlflow.entities import FileInfo
from mlflow.entities.multipart_upload import (
from mlflow.environment_variables import (
from mlflow.exceptions import MlflowException, _UnsupportedMultipartUploadException
from mlflow.store.artifact.artifact_repo import (
from mlflow.store.artifact.cloud_artifact_repo import _complete_futures, _compute_num_chunks
from mlflow.utils.credentials import get_default_host_creds
from mlflow.utils.file_utils import read_chunk, relative_path_to_artifact_path
from mlflow.utils.mime_type_utils import _guess_mime_type
from mlflow.utils.rest_utils import augmented_raise_for_status, http_request
from mlflow.utils.uri import validate_path_is_safe
def _try_multipart_upload(self, local_file, artifact_path=None):
    """
        Attempts to perform multipart upload to log an artifact.
        Returns if the multipart upload is successful.
        Raises UnsupportedMultipartUploadException if multipart upload is unsupported.
        """
    chunk_size = MLFLOW_MULTIPART_UPLOAD_CHUNK_SIZE.get()
    num_parts = _compute_num_chunks(local_file, chunk_size)
    try:
        create = self.create_multipart_upload(local_file, num_parts, artifact_path)
    except HTTPError as e:
        error_message = e.response.json().get('message', '')
        if isinstance(error_message, str) and error_message.startswith(_UnsupportedMultipartUploadException.MESSAGE):
            raise _UnsupportedMultipartUploadException()
        raise
    try:
        futures = {}
        for i, credential in enumerate(create.credentials):
            future = self.thread_pool.submit(self._upload_part, credential=credential, local_file=local_file, size=chunk_size, start_byte=chunk_size * i)
            futures[future] = credential.part_number
        parts, errors = _complete_futures(futures, local_file)
        if errors:
            raise MlflowException(f'Failed to upload at least one part of {local_file}. Errors: {errors}')
        parts = sorted(parts.values(), key=lambda part: part.part_number)
        self.complete_multipart_upload(local_file, create.upload_id, parts, artifact_path)
    except Exception as e:
        self.abort_multipart_upload(local_file, create.upload_id, artifact_path)
        _logger.warning(f'Failed to upload file {local_file} using multipart upload: {e}')
        raise