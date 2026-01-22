import datetime
import importlib.metadata
import os
import posixpath
import urllib.parse
from collections import namedtuple
from packaging.version import Version
from mlflow.entities import FileInfo
from mlflow.entities.multipart_upload import (
from mlflow.environment_variables import (
from mlflow.exceptions import _UnsupportedMultipartUploadException
from mlflow.store.artifact.artifact_repo import ArtifactRepository, MultipartUploadMixin
from mlflow.utils.file_utils import relative_path_to_artifact_path
def _list_folders(self, bkt, prefix, artifact_path):
    results = bkt.list_blobs(prefix=prefix, delimiter='/')
    dir_paths = set()
    for page in results.pages:
        dir_paths.update(page.prefixes)
    return [FileInfo(path[len(artifact_path) + 1:-1], True, None) for path in dir_paths]