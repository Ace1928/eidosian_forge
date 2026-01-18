from __future__ import annotations
import contextlib
import decimal
import warnings
import weakref
import xml.etree.ElementTree
from datetime import datetime, timedelta
from itertools import product
from operator import add
from textwrap import dedent
import numpy as np
import pandas as pd
import pytest
from pandas.errors import PerformanceWarning
from pandas.io.formats import format as pandas_format
import dask
import dask.array as da
import dask.dataframe as dd
import dask.dataframe.groupby
from dask import delayed
from dask.base import compute_as_if_collection
from dask.blockwise import fuse_roots
from dask.dataframe import _compat, methods
from dask.dataframe._compat import (
from dask.dataframe._pyarrow import to_pyarrow_string
from dask.dataframe.core import (
from dask.dataframe.utils import (
from dask.datasets import timeseries
from dask.utils import M, is_dataframe_like, is_series_like, put_lines
from dask.utils_test import _check_warning, hlg_layer
def test_where_mask():
    pdf1 = pd.DataFrame({'a': [1, 2, 3, 4, 5, 6, 7, 8, 9], 'b': [3, 5, 2, 5, 7, 2, 4, 2, 4]})
    ddf1 = dd.from_pandas(pdf1, 2)
    pdf2 = pd.DataFrame({'a': [True, False, True] * 3, 'b': [False, False, True] * 3})
    ddf2 = dd.from_pandas(pdf2, 2)
    pdf3 = pd.DataFrame({'a': [1, 2, 3, 4, 5, 6, 7, 8, 9], 'b': [3, 5, 2, 5, 7, 2, 4, 2, 4]}, index=[0, 1, 2, 3, 4, 5, 6, 7, 8])
    ddf3 = dd.from_pandas(pdf3, 2)
    pdf4 = pd.DataFrame({'a': [True, False, True] * 3, 'b': [False, False, True] * 3}, index=[5, 6, 7, 8, 9, 10, 11, 12, 13])
    ddf4 = dd.from_pandas(pdf4, 2)
    pdf5 = pd.DataFrame({'a': [1, 2, 3, 4, 5, 6, 7, 8, 9], 'b': [9, 4, 2, 6, 2, 3, 1, 6, 2], 'c': [5, 6, 7, 8, 9, 10, 11, 12, 13]}, index=[0, 1, 2, 3, 4, 5, 6, 7, 8])
    ddf5 = dd.from_pandas(pdf5, 2)
    pdf6 = pd.DataFrame({'a': [True, False, True] * 3, 'b': [False, False, True] * 3, 'c': [False] * 9, 'd': [True] * 9}, index=[5, 6, 7, 8, 9, 10, 11, 12, 13])
    ddf6 = dd.from_pandas(pdf6, 2)
    cases = [(ddf1, ddf2, pdf1, pdf2), (ddf1.repartition([0, 3, 6, 8]), ddf2, pdf1, pdf2), (ddf1, ddf4, pdf3, pdf4), (ddf3.repartition([0, 4, 6, 8]), ddf4.repartition([5, 9, 10, 13]), pdf3, pdf4), (ddf5, ddf6, pdf5, pdf6), (ddf5.repartition([0, 4, 7, 8]), ddf6, pdf5, pdf6), (ddf1, pdf2, pdf1, pdf2), (ddf1, pdf4, pdf3, pdf4), (ddf5, pdf6, pdf5, pdf6)]
    for ddf, ddcond, pdf, pdcond in cases:
        assert isinstance(ddf, dd.DataFrame)
        assert isinstance(ddcond, (dd.DataFrame, pd.DataFrame))
        assert isinstance(pdf, pd.DataFrame)
        assert isinstance(pdcond, pd.DataFrame)
        assert_eq(ddf.where(ddcond), pdf.where(pdcond))
        assert_eq(ddf.mask(ddcond), pdf.mask(pdcond))
        assert_eq(ddf.where(ddcond, -ddf), pdf.where(pdcond, -pdf))
        assert_eq(ddf.mask(ddcond, -ddf), pdf.mask(pdcond, -pdf))
        assert_eq(ddf.where(ddcond.a, -ddf), pdf.where(pdcond.a, -pdf))
        assert_eq(ddf.mask(ddcond.a, -ddf), pdf.mask(pdcond.a, -pdf))
        assert_eq(ddf.a.where(ddcond.a), pdf.a.where(pdcond.a))
        assert_eq(ddf.a.mask(ddcond.a), pdf.a.mask(pdcond.a))
        assert_eq(ddf.a.where(ddcond.a, -ddf.a), pdf.a.where(pdcond.a, -pdf.a))
        assert_eq(ddf.a.mask(ddcond.a, -ddf.a), pdf.a.mask(pdcond.a, -pdf.a))