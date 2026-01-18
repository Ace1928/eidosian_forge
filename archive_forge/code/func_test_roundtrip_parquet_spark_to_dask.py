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
def test_roundtrip_parquet_spark_to_dask(spark_session, npartitions, tmpdir, engine):
    tmpdir = str(tmpdir)
    sdf = spark_session.createDataFrame(pdf)
    sdf.repartition(npartitions).write.parquet(tmpdir, mode='overwrite')
    ddf = dd.read_parquet(tmpdir, engine=engine)
    ddf = ddf.assign(timestamp=ddf.timestamp.dt.tz_localize('UTC'))
    assert ddf.npartitions == npartitions
    assert_eq(ddf, pdf, check_index=False)