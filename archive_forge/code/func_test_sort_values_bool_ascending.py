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
def test_sort_values_bool_ascending():
    df = pd.DataFrame({'a': [1, 2, 3] * 20, 'b': [4, 5, 6, 7] * 15})
    ddf = dd.from_pandas(df, npartitions=10)
    with pytest.raises(ValueError, match='length'):
        ddf.sort_values(by='a', ascending=[True, False])
    with pytest.raises(ValueError, match='length'):
        ddf.sort_values(by=['a', 'b'], ascending=[True])
    assert_eq(ddf.sort_values(by='a', ascending=[True]), df.sort_values(by='a', ascending=[True]))
    assert_eq(ddf.sort_values(by=['a', 'b'], ascending=[True, False]), df.sort_values(by=['a', 'b'], ascending=[True, False]))