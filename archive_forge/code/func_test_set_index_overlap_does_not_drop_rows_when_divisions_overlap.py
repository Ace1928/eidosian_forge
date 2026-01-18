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
def test_set_index_overlap_does_not_drop_rows_when_divisions_overlap():
    df = pd.DataFrame({'ts': [1, 1, 2, 2, 3, 3, 3, 3], 'value': 'abc'})
    ddf = dd.from_pandas(df, npartitions=3)
    expected = df.set_index('ts')
    actual = ddf.set_index('ts', sorted=True)
    assert_eq(expected, actual)