from __future__ import annotations
import contextlib
import itertools
import multiprocessing as mp
import os
import pickle
import random
import string
import tempfile
from concurrent.futures import ProcessPoolExecutor
from copy import copy
from datetime import date, time
from decimal import Decimal
from functools import partial
from unittest import mock
import numpy as np
import pandas as pd
import pytest
import dask
import dask.dataframe as dd
from dask.base import compute_as_if_collection
from dask.dataframe._compat import (
from dask.dataframe.shuffle import (
from dask.dataframe.utils import assert_eq, make_meta
from dask.optimization import cull
@pytest.mark.skipif(pa is None, reason='Need pyarrow')
@pytest.mark.skipif(not PANDAS_GE_200, reason='dtype support not good before 2.0')
@pytest.mark.parametrize('data, dtype', [(['a', 'b'], 'string[pyarrow]'), ([b'a', b'b'], 'binary[pyarrow]'), ([1, 2], 'int64[pyarrow]'), ([1, 2], 'float64[pyarrow]'), ([1, 2], 'uint64[pyarrow]'), ([date(2022, 1, 1), date(1999, 12, 31)], 'date32[pyarrow]'), ([pd.Timestamp('2022-01-01'), pd.Timestamp('2023-01-02')], 'timestamp[ns][pyarrow]'), ([Decimal('5'), Decimal('6.24')], 'decimal128'), ([pd.Timedelta('1 day'), pd.Timedelta('20 days')], 'duration[ns][pyarrow]'), ([time(12, 0), time(0, 12)], 'time64[ns][pyarrow]')])
def test_set_index_pyarrow_dtype(data, dtype):
    if dtype == 'decimal128':
        dtype = pd.ArrowDtype(pa.decimal128(10, 2))
    pdf = pd.DataFrame({'a': 1, 'arrow_col': pd.Series(data, dtype=dtype)})
    ddf = dd.from_pandas(pdf, npartitions=2)
    pdf_result = pdf.set_index('arrow_col')
    ddf_result = ddf.set_index('arrow_col')
    assert_eq(ddf_result, pdf_result)