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
def test_set_index_detects_sorted_data(shuffle_method):
    df = pd.DataFrame({'x': range(100), 'y': range(100)})
    if DASK_EXPR_ENABLED:
        ddf = dd.from_pandas(df, npartitions=10, sort=False)
    else:
        ddf = dd.from_pandas(df, npartitions=10, name='x', sort=False)
    ddf2 = ddf.set_index('x', shuffle_method=shuffle_method)
    assert len(ddf2.dask) < ddf.npartitions * 4