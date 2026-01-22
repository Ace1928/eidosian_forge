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
def _get_aggregation_depth(aggregate_files, partition_names):
    aggregation_depth = aggregate_files
    if isinstance(aggregate_files, str):
        if aggregate_files in partition_names:
            aggregation_depth = len(partition_names) - partition_names.index(aggregate_files)
        else:
            raise ValueError(f'{aggregate_files} is not a recognized directory partition.')
    return aggregation_depth