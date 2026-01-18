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
def test_partitioning_index_categorical_on_values():
    df = pd.DataFrame({'a': list(string.ascii_letters), 'b': [1, 2, 3, 4] * 13})
    df.a = df.a.astype('category')
    df2 = df.copy()
    df2.a = df2.a.cat.set_categories(list(reversed(df2.a.cat.categories)))
    res = partitioning_index(df.a, 5)
    res2 = partitioning_index(df2.a, 5)
    assert (res == res2).all()
    res = partitioning_index(df, 5)
    res2 = partitioning_index(df2, 5)
    assert (res == res2).all()