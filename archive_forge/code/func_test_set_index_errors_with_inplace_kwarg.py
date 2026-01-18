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
@pytest.mark.skipif(DASK_EXPR_ENABLED, reason="we don't support inplace")
def test_set_index_errors_with_inplace_kwarg():
    df = pd.DataFrame({'a': [9, 8, 7], 'b': [6, 5, 4], 'c': [3, 2, 1]})
    ddf = dd.from_pandas(df, npartitions=1)
    ddf.set_index('a')
    with pytest.raises(NotImplementedError):
        ddf.set_index('a', inplace=True)