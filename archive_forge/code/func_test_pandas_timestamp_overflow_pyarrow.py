from __future__ import annotations
import contextlib
import glob
import math
import os
import sys
import warnings
from datetime import date
from decimal import Decimal
from unittest.mock import MagicMock
import numpy as np
import pandas as pd
import pytest
from packaging.version import parse as parse_version
import dask
import dask.dataframe as dd
import dask.multiprocessing
from dask.array.numpy_compat import NUMPY_GE_124
from dask.blockwise import Blockwise, optimize_blockwise
from dask.dataframe._compat import (
from dask.dataframe.io.parquet.core import get_engine
from dask.dataframe.io.parquet.utils import _parse_pandas_metadata
from dask.dataframe.optimize import optimize_dataframe_getitem
from dask.dataframe.utils import assert_eq, pyarrow_strings_enabled
from dask.layers import DataFrameIOLayer
from dask.utils import natural_sort_key
from dask.utils_test import hlg_layer
@PYARROW_MARK
def test_pandas_timestamp_overflow_pyarrow(tmpdir):
    info = np.iinfo(np.dtype('int64'))
    if NUMPY_GE_124:
        ctx = pytest.warns(RuntimeWarning, match='invalid value encountered in cast')
    else:
        ctx = contextlib.nullcontext()
    with ctx:
        arr_numeric = np.linspace(start=info.min + 2, stop=info.max, num=1024, dtype='int64')
    arr_dates = arr_numeric.astype('datetime64[ms]')
    table = pa.Table.from_arrays([pa.array(arr_dates)], names=['ts'])
    pa.parquet.write_table(table, f'{tmpdir}/file.parquet', use_deprecated_int96_timestamps=False)
    if pyarrow_version < parse_version('13.0.0.dev'):
        with pytest.raises(pa.lib.ArrowInvalid) as e:
            dd.read_parquet(str(tmpdir)).compute()
            assert 'out of bounds' in str(e.value)
    else:
        dd.read_parquet(str(tmpdir)).compute()
    from dask.dataframe.io.parquet.arrow import ArrowDatasetEngine

    class ArrowEngineWithTimestampClamp(ArrowDatasetEngine):

        @classmethod
        def clamp_arrow_datetimes(cls, arrow_table: pa.Table) -> pa.Table:
            """Constrain datetimes to be valid for pandas

            Since pandas works in ns precision and arrow / parquet defaults to ms
            precision we need to clamp our datetimes to something reasonable"""
            new_columns = []
            for col in arrow_table.columns:
                if pa.types.is_timestamp(col.type) and col.type.unit in ('s', 'ms', 'us'):
                    multiplier = {'s': 10000000000, 'ms': 1000000, 'us': 1000}[col.type.unit]
                    original_type = col.type
                    series: pd.Series = col.cast(pa.int64()).to_pandas()
                    info = np.iinfo(np.dtype('int64'))
                    series.clip(lower=info.min // multiplier + 1, upper=info.max // multiplier, inplace=True)
                    new_array = pa.array(series, pa.int64())
                    new_array = new_array.cast(original_type)
                    new_columns.append(new_array)
                else:
                    new_columns.append(col)
            return pa.Table.from_arrays(new_columns, names=arrow_table.column_names)

        @classmethod
        def _arrow_table_to_pandas(cls, arrow_table: pa.Table, categories, dtype_backend=None, convert_string=False, **kwargs) -> pd.DataFrame:
            fixed_arrow_table = cls.clamp_arrow_datetimes(arrow_table)
            return super()._arrow_table_to_pandas(fixed_arrow_table, categories, dtype_backend=dtype_backend, convert_string=convert_string, **kwargs)
    dd.read_parquet(str(tmpdir), engine=ArrowEngineWithTimestampClamp).compute()