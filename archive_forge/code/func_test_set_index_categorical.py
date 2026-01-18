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
def test_set_index_categorical():
    order = list(reversed(string.ascii_letters))
    values = list(string.ascii_letters)
    random.shuffle(values)
    dtype = pd.api.types.CategoricalDtype(order, ordered=True)
    df = pd.DataFrame({'A': pd.Categorical(values, dtype=dtype), 'B': 1})
    result = dd.from_pandas(df, npartitions=2).set_index('A')
    assert len(result) == len(df)
    divisions = pd.Categorical(result.divisions, dtype=dtype)
    assert_categorical_equal(divisions, divisions.sort_values())