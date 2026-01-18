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
@pytest.mark.parametrize('npartitions', [1, 5, 10])
def test_roundtrip_parquet_dask_to_spark(spark_session, npartitions, tmpdir, engine):
    tmpdir = str(tmpdir)
    ddf = dd.from_pandas(pdf, npartitions=npartitions)
    kwargs = {'times': 'int96'} if engine == 'fastparquet' else {}
    ddf.to_parquet(tmpdir, engine=engine, write_index=False, **kwargs)
    sdf = spark_session.read.parquet(tmpdir)
    sdf = sdf.toPandas()
    sdf = sdf.assign(timestamp=sdf.timestamp.dt.tz_localize('UTC'))
    assert_eq(sdf, ddf, check_index=False)