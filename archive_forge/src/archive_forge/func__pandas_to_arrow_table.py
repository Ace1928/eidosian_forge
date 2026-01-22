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
def _pandas_to_arrow_table(cls, df: pd.DataFrame, preserve_index=False, schema=None) -> pa.Table:
    try:
        return pa.Table.from_pandas(df, nthreads=1, preserve_index=preserve_index, schema=schema)
    except pa.ArrowException as exc:
        if schema is None:
            raise
        df_schema = pa.Schema.from_pandas(df)
        expected = textwrap.indent(schema.to_string(show_schema_metadata=False), '    ')
        actual = textwrap.indent(df_schema.to_string(show_schema_metadata=False), '    ')
        raise ValueError(f'Failed to convert partition to expected pyarrow schema:\n    `{exc!r}`\n\nExpected partition schema:\n{expected}\n\nReceived partition schema:\n{actual}\n\nThis error *may* be resolved by passing in schema information for\nthe mismatched column(s) using the `schema` keyword in `to_parquet`.') from None