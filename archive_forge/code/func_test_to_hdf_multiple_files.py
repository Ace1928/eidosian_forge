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
def test_to_hdf_multiple_files():
    pytest.importorskip('tables')
    df = pd.DataFrame({'x': ['a', 'b', 'c', 'd'], 'y': [1, 2, 3, 4]}, index=[1.0, 2.0, 3.0, 4.0])
    a = dd.from_pandas(df, 2)
    df16 = pd.DataFrame({'x': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p'], 'y': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]}, index=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0])
    b = dd.from_pandas(df16, 16)
    with tmpdir() as dn:
        fn = os.path.join(dn, 'data_*.h5')
        a.to_hdf(fn, '/data')
        out = dd.read_hdf(fn, '/data')
        assert_eq(df, out)
    with tmpdir() as dn:
        fn = os.path.join(dn, 'data_*.h5')
        b.to_hdf(fn, '/data')
        out = dd.read_hdf(fn, '/data')
        assert_eq(df16, out)
    with tmpdir() as dn:
        fn1 = os.path.join(dn, 'data_1.h5')
        fn2 = os.path.join(dn, 'data_2.h5')
        b.to_hdf(fn1, '/data')
        a.to_hdf(fn2, '/data')
        out = dd.read_hdf([fn1, fn2], '/data')
        assert_eq(pd.concat([df16, df]), out)
    with tmpdir() as dn:
        fn = os.path.join(dn, 'data_*.h5')
        a.to_hdf(fn, '/data', name_function=lambda i: 'a' * (i + 1))
        out = dd.read_hdf(fn, '/data')
        assert_eq(df, out)
        out = pd.read_hdf(os.path.join(dn, 'data_a.h5'), '/data')
        tm.assert_frame_equal(out, df.iloc[:2])
        out = pd.read_hdf(os.path.join(dn, 'data_aa.h5'), '/data')
        tm.assert_frame_equal(out, df.iloc[2:])
    with tmpfile('h5') as fn:
        with pd.HDFStore(fn) as hdf:
            a.to_hdf(hdf, '/data*')
        out = dd.read_hdf(fn, '/data*')
        assert_eq(df, out)