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
@pytest.mark.skipif(not PANDAS_GE_140, reason='EA Indexes not supported before')
def test_set_index_ea_dtype():
    pdf = pd.DataFrame({'a': 1, 'b': pd.Series([1, 2], dtype='Int64')})
    ddf = dd.from_pandas(pdf, npartitions=2)
    pdf_result = pdf.set_index('b')
    ddf_result = ddf.set_index('b')
    assert_eq(ddf_result, pdf_result)