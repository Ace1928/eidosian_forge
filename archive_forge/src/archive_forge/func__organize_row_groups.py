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
def _organize_row_groups(cls, pf, split_row_groups, gather_statistics, stat_col_indices, filters, dtypes, base_path, has_metadata_file, blocksize, aggregation_depth):
    """Organize row-groups by file."""
    pqpartitions = list(pf.cats)
    if pqpartitions and aggregation_depth and pf.row_groups and pf.row_groups[0].columns[0].file_path:
        pf.row_groups = sorted(pf.row_groups, key=lambda x: natural_sort_key(x.columns[0].file_path))
    pandas_type = {}
    if pf.row_groups and pf.pandas_metadata:
        for c in pf.pandas_metadata.get('columns', []):
            if 'field_name' in c:
                pandas_type[c['field_name']] = c.get('pandas_type', None)
    single_rg_parts = int(split_row_groups) == 1
    file_row_groups = defaultdict(list)
    file_row_group_stats = defaultdict(list)
    file_row_group_column_stats = defaultdict(list)
    cmax_last = {}
    for rg, row_group in enumerate(pf.row_groups):
        if pqpartitions and filters and fastparquet.api.filter_out_cats(row_group, filters):
            continue
        fp = row_group.columns[0].file_path
        fpath = fp.decode() if isinstance(fp, bytes) else fp
        if fpath is None:
            if not has_metadata_file:
                fpath = pf.fn
                base_path = base_path or ''
            else:
                raise ValueError('Global metadata structure is missing a file_path string. If the dataset includes a _metadata file, that file may have one or more missing file_path fields.')
        if file_row_groups[fpath]:
            file_row_groups[fpath].append((file_row_groups[fpath][-1][0] + 1, rg))
        else:
            file_row_groups[fpath].append((0, rg))
        if gather_statistics:
            if single_rg_parts:
                s = {'file_path_0': fpath, 'num-rows': row_group.num_rows, 'total_byte_size': row_group.total_byte_size, 'columns': []}
            else:
                s = {'num-rows': row_group.num_rows, 'total_byte_size': row_group.total_byte_size}
            cstats = []
            for name, i in stat_col_indices.items():
                column = row_group.columns[i]
                if column.meta_data.statistics:
                    cmin = None
                    cmax = None
                    null_count = None
                    if pf.statistics['min'][name][0] is not None:
                        cmin = pf.statistics['min'][name][rg]
                        cmax = pf.statistics['max'][name][rg]
                        null_count = pf.statistics['null_count'][name][rg]
                    elif dtypes[name] == 'object':
                        cmin = column.meta_data.statistics.min_value
                        cmax = column.meta_data.statistics.max_value
                        null_count = column.meta_data.statistics.null_count
                        if cmin is None:
                            cmin = column.meta_data.statistics.min
                        if cmax is None:
                            cmax = column.meta_data.statistics.max
                        if isinstance(cmin, (bytes, bytearray)) and pandas_type.get(name, None) != 'bytes':
                            cmin = cmin.decode('utf-8')
                            cmax = cmax.decode('utf-8')
                        if isinstance(null_count, (bytes, bytearray)):
                            null_count = null_count.decode('utf-8')
                    if isinstance(cmin, np.datetime64):
                        tz = getattr(dtypes[name], 'tz', None)
                        cmin = pd.Timestamp(cmin, tz=tz)
                        cmax = pd.Timestamp(cmax, tz=tz)
                    last = cmax_last.get(name, None)
                    if not (filters or (blocksize and split_row_groups is True) or aggregation_depth):
                        if cmin is None or (last and cmin < last):
                            gather_statistics = False
                            file_row_group_stats = {}
                            file_row_group_column_stats = {}
                            break
                    if single_rg_parts:
                        s['columns'].append({'name': name, 'min': cmin, 'max': cmax, 'null_count': null_count})
                    else:
                        cstats += [cmin, cmax, null_count]
                    cmax_last[name] = cmax
                else:
                    if not (filters or (blocksize and split_row_groups is True) or aggregation_depth) and column.meta_data.num_values > 0:
                        gather_statistics = False
                        file_row_group_stats = {}
                        file_row_group_column_stats = {}
                        break
                    if single_rg_parts:
                        s['columns'].append({'name': name})
                    else:
                        cstats += [None, None, None]
            if gather_statistics:
                file_row_group_stats[fpath].append(s)
                if not single_rg_parts:
                    file_row_group_column_stats[fpath].append(tuple(cstats))
    return (file_row_groups, file_row_group_stats, file_row_group_column_stats, gather_statistics, base_path)