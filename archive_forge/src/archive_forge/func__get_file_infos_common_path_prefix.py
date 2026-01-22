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
def _get_file_infos_common_path_prefix(paths: List[str], common_path: str, filesystem: 'pyarrow.fs.FileSystem', ignore_missing_paths: bool=False) -> Iterator[Tuple[str, int]]:
    path_to_size = {path: None for path in paths}
    for path, file_size in _get_file_infos(common_path, filesystem, ignore_missing_paths):
        if path in path_to_size:
            path_to_size[path] = file_size
    have_missing_path = False
    for path in paths:
        if path_to_size[path] is None:
            logger.debug(f'Finding path {path} not have file size metadata. Fall back to get files metadata in parallel for all paths.')
            have_missing_path = True
            break
    if have_missing_path:
        yield from _get_file_infos_parallel(paths, filesystem, ignore_missing_paths)
    else:
        for path in paths:
            yield (path, path_to_size[path])