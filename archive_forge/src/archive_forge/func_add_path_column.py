from __future__ import annotations
import io
import os
from functools import partial
from itertools import zip_longest
import pandas as pd
from fsspec.core import open_files
from dask.base import compute as dask_compute
from dask.bytes import read_bytes
from dask.core import flatten
from dask.dataframe._compat import PANDAS_GE_200, PANDAS_VERSION
from dask.dataframe.backends import dataframe_creation_dispatch
from dask.dataframe.io.io import from_delayed
from dask.dataframe.utils import insert_meta_param_description, make_meta
from dask.delayed import delayed
def add_path_column(df, column_name, path, dtype):
    if column_name in df.columns:
        raise ValueError(f"Files already contain the column name: '{column_name}', so the path column cannot use this name. Please set `include_path_column` to a unique name.")
    return df.assign(**{column_name: pd.Series([path] * len(df), dtype=dtype)})