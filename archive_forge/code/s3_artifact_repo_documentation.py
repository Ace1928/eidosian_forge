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
Parse an S3 URI, returning (bucket, path)