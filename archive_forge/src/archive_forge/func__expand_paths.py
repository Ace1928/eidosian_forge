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
def _expand_paths(paths: List[str], filesystem: 'pyarrow.fs.FileSystem', partitioning: Optional[Partitioning], ignore_missing_paths: bool=False) -> Iterator[Tuple[str, int]]:
    """Get the file sizes for all provided file paths."""
    from pyarrow.fs import LocalFileSystem
    from ray.data.datasource.file_based_datasource import FILE_SIZE_FETCH_PARALLELIZATION_THRESHOLD
    from ray.data.datasource.path_util import _unwrap_protocol
    if len(paths) < FILE_SIZE_FETCH_PARALLELIZATION_THRESHOLD or isinstance(filesystem, LocalFileSystem):
        yield from _get_file_infos_serial(paths, filesystem, ignore_missing_paths)
    else:
        common_path = os.path.commonpath(paths)
        if partitioning is not None and common_path == _unwrap_protocol(partitioning.base_dir) or all((str(pathlib.Path(path).parent) == common_path for path in paths)):
            yield from _get_file_infos_common_path_prefix(paths, common_path, filesystem, ignore_missing_paths)
        else:
            yield from _get_file_infos_parallel(paths, filesystem, ignore_missing_paths)