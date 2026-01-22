import atexit
import codecs
import errno
import fnmatch
import gzip
import json
import logging
import math
import os
import pathlib
import posixpath
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.parse
import urllib.request
import uuid
from concurrent.futures import as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from subprocess import CalledProcessError, TimeoutExpired
from typing import Optional, Union
from urllib.parse import unquote
from urllib.request import pathname2url
import yaml
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.entities import FileInfo
from mlflow.environment_variables import (
from mlflow.exceptions import MissingConfigException, MlflowException
from mlflow.protos.databricks_artifacts_pb2 import ArtifactCredentialType
from mlflow.utils import download_cloud_file_chunk, merge_dicts
from mlflow.utils.databricks_utils import _get_dbutils
from mlflow.utils.os import is_windows
from mlflow.utils.process import cache_return_value_per_process
from mlflow.utils.request_utils import cloud_storage_http_request, download_chunk
from mlflow.utils.rest_utils import augmented_raise_for_status
def _handle_readonly_on_windows(func, path, exc_info):
    """
    This function should not be called directly but should be passed to `onerror` of
    `shutil.rmtree` in order to reattempt the removal of a read-only file after making
    it writable on Windows.

    References:
    - https://bugs.python.org/issue19643
    - https://bugs.python.org/issue43657
    """
    exc_type, exc_value = exc_info[:2]
    should_reattempt = is_windows() and func in (os.unlink, os.rmdir) and issubclass(exc_type, PermissionError) and (exc_value.winerror == 5)
    if not should_reattempt:
        raise exc_value
    os.chmod(path, stat.S_IWRITE)
    func(path)