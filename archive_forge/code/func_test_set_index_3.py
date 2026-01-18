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
def test_set_index_3(shuffle_method):
    df = pd.DataFrame(np.random.random((10, 2)), columns=['x', 'y'])
    ddf = dd.from_pandas(df, npartitions=5)
    ddf2 = ddf.set_index('x', shuffle_method=shuffle_method, max_branch=2, npartitions=ddf.npartitions)
    df2 = df.set_index('x')
    assert_eq(df2, ddf2)
    assert ddf2.npartitions == ddf.npartitions