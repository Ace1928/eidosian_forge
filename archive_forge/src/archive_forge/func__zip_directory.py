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
def _zip_directory(directory: str, excludes: List[str], output_path: str, include_parent_dir: bool=False, logger: Optional[logging.Logger]=default_logger) -> None:
    """Zip the target directory and write it to the output_path.

    directory: The directory to zip.
    excludes (List(str)): The directories or file to be excluded.
    output_path: The output path for the zip file.
    include_parent_dir: If true, includes the top-level directory as a
        directory inside the zip file.
    """
    pkg_file = Path(output_path).absolute()
    with ZipFile(pkg_file, 'w') as zip_handler:
        dir_path = Path(directory).absolute()

        def handler(path: Path):
            if path.is_dir() and next(path.iterdir(), None) is None or path.is_file():
                file_size = path.stat().st_size
                if file_size >= FILE_SIZE_WARNING:
                    logger.warning(f"File {path} is very large ({_mib_string(file_size)}). Consider adding this file to the 'excludes' list to skip uploading it: `ray.init(..., runtime_env={{'excludes': ['{path}']}})`")
                to_path = path.relative_to(dir_path)
                if include_parent_dir:
                    to_path = dir_path.name / to_path
                zip_handler.write(path, to_path)
        excludes = [_get_excludes(dir_path, excludes)]
        _dir_travel(dir_path, excludes, handler, logger=logger)