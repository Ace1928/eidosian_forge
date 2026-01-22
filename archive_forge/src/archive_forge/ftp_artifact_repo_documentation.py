import ftplib
import os
import posixpath
import urllib.parse
from contextlib import contextmanager
from ftplib import FTP
from urllib.parse import unquote
from mlflow.entities.file_info import FileInfo
from mlflow.exceptions import MlflowException
from mlflow.store.artifact.artifact_repo import ArtifactRepository
from mlflow.utils.file_utils import relative_path_to_artifact_path
Stores artifacts as files in a remote directory, via ftp.