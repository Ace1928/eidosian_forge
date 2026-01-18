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
def test_set_index_raises_error_on_bad_input():
    df = pd.DataFrame({'a': [1, 2, 3, 4, 5, 6, 7], 'b': [7, 6, 5, 4, 3, 2, 1]})
    ddf = dd.from_pandas(df, 2)
    msg = 'Dask dataframe does not yet support multi-indexes'
    with pytest.raises(NotImplementedError) as err:
        ddf.set_index(['a', 'b'])
    assert msg in str(err.value)
    with pytest.raises(NotImplementedError) as err:
        ddf.set_index([['a', 'b']])
    assert msg in str(err.value)
    with pytest.raises(NotImplementedError) as err:
        ddf.set_index([['a']])
    assert msg in str(err.value)