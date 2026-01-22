from __future__ import annotations
import json
import operator
import textwrap
import warnings
from collections import defaultdict
from datetime import datetime
from functools import reduce
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.fs as pa_fs
import pyarrow.parquet as pq
from fsspec.core import expand_paths_if_needed, stringify_path
from fsspec.implementations.arrow import ArrowFSWrapper
from pyarrow import dataset as pa_ds
from pyarrow import fs as pa_fs
import dask
from dask.base import normalize_token, tokenize
from dask.core import flatten
from dask.dataframe._compat import PANDAS_GE_220
from dask.dataframe.backends import pyarrow_schema_dispatch
from dask.dataframe.io.parquet.utils import (
from dask.dataframe.io.utils import _get_pyarrow_dtypes, _is_local_fs, _open_input_files
from dask.dataframe.utils import clear_known_categories, pyarrow_strings_enabled
from dask.delayed import Delayed
from dask.utils import getargspec, natural_sort_key
@classmethod
def _read_table(cls, path_or_frag, fs, row_groups, columns, schema, filters, partitions, partition_keys, **kwargs):
    """Read in a pyarrow table"""
    if isinstance(path_or_frag, pa_ds.ParquetFileFragment):
        frag = path_or_frag
    else:
        frag = None
        partitioning = kwargs.get('dataset', {}).get('partitioning', None)
        missing_partitioning_info = partitions and partition_keys is None or (partitioning and (not isinstance(partitioning, (str, list))))
        if missing_partitioning_info or _need_filtering(filters, partition_keys):
            ds = pa_ds.dataset(path_or_frag, filesystem=_wrapped_fs(fs), **_process_kwargs(**kwargs.get('dataset', {})))
            frags = list(ds.get_fragments())
            assert len(frags) == 1
            frag = _frag_subset(frags[0], row_groups) if row_groups != [None] else frags[0]
            raw_keys = pa_ds._get_partition_keys(frag.partition_expression)
            partition_keys = [(hive_part.name, raw_keys[hive_part.name]) for hive_part in partitions]
    if frag:
        cols = []
        for name in columns:
            if name is None:
                if '__index_level_0__' in schema.names:
                    columns.append('__index_level_0__')
            else:
                cols.append(name)
        arrow_table = frag.to_table(use_threads=False, schema=schema, columns=cols, filter=_filters_to_expression(filters) if filters else None)
    else:
        arrow_table = _read_table_from_path(path_or_frag, fs, row_groups, columns, schema, filters, **kwargs)
    if partitions and isinstance(partitions, list):
        keys_dict = {k: v for k, v in partition_keys}
        for partition in partitions:
            if partition.name not in arrow_table.schema.names:
                cat = keys_dict.get(partition.name, None)
                if not len(partition.keys):
                    arr = pa.array(np.full(len(arrow_table), cat))
                else:
                    cat_ind = np.full(len(arrow_table), partition.keys.get_loc(cat), dtype='i4')
                    arr = pa.DictionaryArray.from_arrays(cat_ind, pa.array(partition.keys))
                arrow_table = arrow_table.append_column(partition.name, arr)
    return arrow_table