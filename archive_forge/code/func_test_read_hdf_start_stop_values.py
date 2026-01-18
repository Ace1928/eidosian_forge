from __future__ import annotations
import os
import pathlib
from time import sleep
import numpy as np
import pandas as pd
import pytest
from packaging.version import Version
import dask
import dask.dataframe as dd
from dask._compatibility import PY_VERSION
from dask.dataframe._compat import tm
from dask.dataframe.optimize import optimize_dataframe_getitem
from dask.dataframe.utils import assert_eq
from dask.layers import DataFrameIOLayer
from dask.utils import dependency_depth, tmpdir, tmpfile
def test_read_hdf_start_stop_values():
    pytest.importorskip('tables')
    df = pd.DataFrame({'x': ['a', 'b', 'c', 'd'], 'y': [1, 2, 3, 4]}, index=[1.0, 2.0, 3.0, 4.0])
    with tmpfile('h5') as fn:
        df.to_hdf(fn, key='/data', format='table')
        with pytest.raises(ValueError, match='number of rows'):
            dd.read_hdf(fn, '/data', stop=10)
        with pytest.raises(ValueError, match='is above or equal to'):
            dd.read_hdf(fn, '/data', start=10)
        with pytest.raises(ValueError, match='positive integer'):
            dd.read_hdf(fn, '/data', chunksize=-1)