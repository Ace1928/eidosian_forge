import itertools
import logging
import os
import pathlib
import re
from typing import (
import numpy as np
from ray.data._internal.progress_bar import ProgressBar
from ray.data._internal.remote_fn import cached_remote_fn
from ray.data.block import BlockMetadata
from ray.data.datasource.partitioning import Partitioning
from ray.util.annotations import DeveloperAPI
def _get_file_infos(path: str, filesystem: 'pyarrow.fs.FileSystem', ignore_missing_path: bool=False) -> List[Tuple[str, int]]:
    """Get the file info for all files at or under the provided path."""
    from pyarrow.fs import FileType
    file_infos = []
    try:
        file_info = filesystem.get_file_info(path)
    except OSError as e:
        _handle_read_os_error(e, path)
    if file_info.type == FileType.Directory:
        for file_path, file_size in _expand_directory(path, filesystem):
            file_infos.append((file_path, file_size))
    elif file_info.type == FileType.File:
        file_infos.append((path, file_info.size))
    elif file_info.type == FileType.NotFound and ignore_missing_path:
        pass
    else:
        raise FileNotFoundError(path)
    return file_infos