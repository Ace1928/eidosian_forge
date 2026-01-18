from __future__ import annotations
import decimal
import signal
import sys
import threading
import pytest
from dask.datasets import timeseries
import numpy as np
import pandas as pd
from dask.dataframe._compat import PANDAS_GE_150, PANDAS_GE_200
from dask.dataframe.utils import assert_eq
def test_roundtrip_parquet_spark_to_dask_extension_dtypes(spark_session, tmpdir):
    tmpdir = str(tmpdir)
    npartitions = 5
    size = 20
    pdf = pd.DataFrame({'a': range(size), 'b': np.random.random(size=size), 'c': [True, False] * (size // 2), 'd': ['alice', 'bob'] * (size // 2)})
    pdf = pdf.astype({'a': 'Int64', 'b': 'Float64', 'c': 'boolean', 'd': 'string'})
    assert all([pd.api.types.is_extension_array_dtype(dtype) for dtype in pdf.dtypes])
    sdf = spark_session.createDataFrame(pdf)
    sdf.repartition(npartitions).write.parquet(tmpdir, mode='overwrite')
    ddf = dd.read_parquet(tmpdir, engine='pyarrow', dtype_backend='numpy_nullable')
    assert all([pd.api.types.is_extension_array_dtype(dtype) for dtype in ddf.dtypes]), ddf.dtypes
    assert_eq(ddf, pdf, check_index=False)