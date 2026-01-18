import asyncio
import hashlib
import logging
import os
import shutil
from enum import Enum
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable, List, Optional, Tuple
from urllib.parse import urlparse
from zipfile import ZipFile
from filelock import FileLock
from ray.util.annotations import DeveloperAPI
from ray._private.ray_constants import (
from ray._private.runtime_env.conda_utils import exec_cmd_stream_to_logger
from ray._private.thirdparty.pathspec import PathSpec
from ray.experimental.internal_kv import (
def upload_package_if_needed(pkg_uri: str, base_directory: str, directory: str, include_parent_dir: bool=False, excludes: Optional[List[str]]=None, logger: Optional[logging.Logger]=default_logger) -> bool:
    """Upload the contents of the directory under the given URI.

    This will first create a temporary zip file under the passed
    base_directory.

    If the package already exists in storage, this is a no-op.

    Args:
        pkg_uri: URI of the package to upload.
        base_directory: Directory where package files are stored.
        directory: Directory to be uploaded.
        include_parent_dir: If true, includes the top-level directory as a
            directory inside the zip file.
        excludes: List specifying files to exclude.

    Raises:
        RuntimeError: If the upload fails.
        ValueError: If the pkg_uri is a remote path or if the data's
            size exceeds GCS_STORAGE_MAX_SIZE.
        NotImplementedError: If the protocol of the URI is not supported.
    """
    if excludes is None:
        excludes = []
    if logger is None:
        logger = default_logger
    pin_runtime_env_uri(pkg_uri)
    if package_exists(pkg_uri):
        return False
    package_file = Path(_get_local_path(base_directory, pkg_uri))
    create_package(directory, package_file, include_parent_dir=include_parent_dir, excludes=excludes)
    upload_package_to_gcs(pkg_uri, package_file.read_bytes())
    package_file.unlink()
    return True