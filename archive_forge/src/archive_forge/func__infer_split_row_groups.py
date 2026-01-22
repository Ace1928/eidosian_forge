from __future__ import annotations
import re
import warnings
import numpy as np
import pandas as pd
from fsspec.core import expand_paths_if_needed, get_fs_token_paths, stringify_path
from fsspec.spec import AbstractFileSystem
from dask import config
from dask.dataframe.io.utils import _is_local_fs
from dask.utils import natural_sort_key, parse_bytes
def _infer_split_row_groups(row_group_sizes, blocksize, aggregate_files=False):
    if row_group_sizes:
        blocksize = parse_bytes(blocksize)
        if aggregate_files or np.sum(row_group_sizes) > 2 * blocksize:
            return 'adaptive'
    return False