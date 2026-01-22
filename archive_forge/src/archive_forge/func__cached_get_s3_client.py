import json
import os
import posixpath
import urllib.parse
from datetime import datetime
from functools import lru_cache
from mimetypes import guess_type
from mlflow.entities import FileInfo
from mlflow.entities.multipart_upload import (
from mlflow.environment_variables import (
from mlflow.exceptions import MlflowException
from mlflow.store.artifact.artifact_repo import (
from mlflow.utils.file_utils import relative_path_to_artifact_path
@lru_cache(maxsize=64)
def _cached_get_s3_client(signature_version, addressing_style, s3_endpoint_url, verify, timestamp, access_key_id=None, secret_access_key=None, session_token=None, region_name=None):
    """Returns a boto3 client, caching to avoid extra boto3 verify calls.

    This method is outside of the S3ArtifactRepository as it is
    agnostic and could be used by other instances.

    `maxsize` set to avoid excessive memory consumption in the case
    a user has dynamic endpoints (intentionally or as a bug).

    Some of the boto3 endpoint urls, in very edge cases, might expire
    after twelve hours as that is the current expiration time. To ensure
    we throw an error on verification instead of using an expired endpoint
    we utilise the `timestamp` parameter to invalidate cache.
    """
    import boto3
    from botocore.client import Config
    if signature_version.lower() == 'unsigned':
        from botocore import UNSIGNED
        signature_version = UNSIGNED
    return boto3.client('s3', config=Config(signature_version=signature_version, s3={'addressing_style': addressing_style}), endpoint_url=s3_endpoint_url, verify=verify, aws_access_key_id=access_key_id, aws_secret_access_key=secret_access_key, aws_session_token=session_token, region_name=region_name)