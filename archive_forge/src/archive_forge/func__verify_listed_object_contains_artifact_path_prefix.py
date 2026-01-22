import json
import logging
import math
import os
import posixpath
import urllib.parse
from mimetypes import guess_type
from mlflow.entities import FileInfo
from mlflow.environment_variables import (
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_artifacts_pb2 import ArtifactCredentialInfo
from mlflow.store.artifact.cloud_artifact_repo import (
from mlflow.store.artifact.s3_artifact_repo import _get_s3_client
from mlflow.utils.file_utils import read_chunk
from mlflow.utils.request_utils import cloud_storage_http_request
from mlflow.utils.rest_utils import augmented_raise_for_status
@staticmethod
def _verify_listed_object_contains_artifact_path_prefix(listed_object_path, artifact_path):
    if not listed_object_path.startswith(artifact_path):
        raise MlflowException(f'The path of the listed S3 object does not begin with the specified artifact path. Artifact path: {artifact_path}. Object path: {listed_object_path}.')