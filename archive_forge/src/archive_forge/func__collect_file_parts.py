from __future__ import annotations
import copy
import pickle
import threading
import warnings
from collections import OrderedDict, defaultdict
from contextlib import ExitStack
import numpy as np
import pandas as pd
import tlz as toolz
from packaging.version import parse as parse_version
from dask.core import flatten
from dask.dataframe._compat import PANDAS_GE_201
from dask.base import tokenize
from dask.dataframe.io.parquet.utils import (
from dask.dataframe.io.utils import _is_local_fs, _meta_from_dtypes, _open_input_files
from dask.dataframe.utils import UNKNOWN_CATEGORIES
from dask.delayed import Delayed
from dask.utils import natural_sort_key
@classmethod
def _collect_file_parts(cls, pf_or_files, dataset_info_kwargs):
    fs = dataset_info_kwargs['fs']
    split_row_groups = dataset_info_kwargs['split_row_groups']
    gather_statistics = dataset_info_kwargs['gather_statistics']
    stat_col_indices = dataset_info_kwargs['stat_col_indices']
    filters = dataset_info_kwargs['filters']
    dtypes = dataset_info_kwargs['dtypes']
    blocksize = dataset_info_kwargs['blocksize']
    aggregation_depth = dataset_info_kwargs['aggregation_depth']
    base_path = dataset_info_kwargs.get('base_path', None)
    root_cats = dataset_info_kwargs.get('root_cats', None)
    root_file_scheme = dataset_info_kwargs.get('root_file_scheme', None)
    has_metadata_file = dataset_info_kwargs['has_metadata_file']
    if not isinstance(pf_or_files, fastparquet.api.ParquetFile):
        pf = ParquetFile(pf_or_files, open_with=fs.open, root=base_path)
        pf.cats = root_cats or {}
        if root_cats:
            pf.file_scheme = root_file_scheme
    else:
        pf = pf_or_files
    file_row_groups, file_row_group_stats, file_row_group_column_stats, gather_statistics, base_path = cls._organize_row_groups(pf, split_row_groups, gather_statistics, stat_col_indices, filters, dtypes, base_path, has_metadata_file, blocksize, aggregation_depth)
    parts, stats = _row_groups_to_parts(gather_statistics, split_row_groups, aggregation_depth, file_row_groups, file_row_group_stats, file_row_group_column_stats, stat_col_indices, cls._make_part, make_part_kwargs={'fs': fs, 'pf': pf, 'base_path': base_path, 'partitions': list(pf.cats)})
    return (parts, stats)