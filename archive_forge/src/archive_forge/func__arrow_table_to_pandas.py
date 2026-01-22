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
def _arrow_table_to_pandas(cls, arrow_table: pa.Table, categories, dtype_backend=None, convert_string=False, **kwargs) -> pd.DataFrame:
    _kwargs = kwargs.get('arrow_to_pandas', {})
    _kwargs.update({'use_threads': False, 'ignore_metadata': False})
    types_mapper = cls._determine_type_mapper(dtype_backend=dtype_backend, convert_string=convert_string, **kwargs)
    if types_mapper is not None:
        _kwargs['types_mapper'] = types_mapper
    res = arrow_table.to_pandas(categories=categories, **_kwargs)
    if convert_string and isinstance(res.index, pd.Index) and (not isinstance(res.index, pd.MultiIndex)) and pd.api.types.is_string_dtype(res.index.dtype) and (res.index.dtype not in (pd.StringDtype('pyarrow'), pd.ArrowDtype(pa.string()))):
        res.index = res.index.astype(pd.StringDtype('pyarrow'))
    return res