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
def unzip_package(package_path: str, target_dir: str, remove_top_level_directory: bool, unlink_zip: bool, logger: Optional[logging.Logger]=default_logger) -> None:
    """
    Unzip the compressed package contained at package_path to target_dir.

    If remove_top_level_directory is True and the top level consists of a
    a single directory (or possibly also a second hidden directory named
    __MACOSX at the top level arising from macOS's zip command), the function
    will automatically remove the top-level directory and store the contents
    directly in target_dir.

    Otherwise, if remove_top_level_directory is False or if the top level
    consists of multiple files or directories (not counting __MACOS),
    the zip contents will be stored in target_dir.

    Args:
        package_path: String path of the compressed package to unzip.
        target_dir: String path of the directory to store the unzipped contents.
        remove_top_level_directory: Whether to remove the top-level directory
            from the zip contents.
        unlink_zip: Whether to unlink the zip file stored at package_path.
        logger: Optional logger to use for logging.

    """
    try:
        os.mkdir(target_dir)
    except FileExistsError:
        logger.info(f'Directory at {target_dir} already exists')
    logger.debug(f'Unpacking {package_path} to {target_dir}')
    with ZipFile(str(package_path), 'r') as zip_ref:
        zip_ref.extractall(target_dir)
    if remove_top_level_directory:
        top_level_directory = get_top_level_dir_from_compressed_package(package_path)
        if top_level_directory is not None:
            macos_dir = os.path.join(target_dir, MAC_OS_ZIP_HIDDEN_DIR_NAME)
            if os.path.isdir(macos_dir):
                shutil.rmtree(macos_dir)
            remove_dir_from_filepaths(target_dir, top_level_directory)
    if unlink_zip:
        Path(package_path).unlink()