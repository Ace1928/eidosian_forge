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
@property
def _host_creds(self):
    return get_default_host_creds(self.artifact_uri)