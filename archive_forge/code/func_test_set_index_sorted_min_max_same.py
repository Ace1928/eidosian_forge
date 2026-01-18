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
def test_set_index_sorted_min_max_same():
    a = pd.DataFrame({'x': [1, 2, 3], 'y': [0, 0, 0]})
    b = pd.DataFrame({'x': [1, 2, 3], 'y': [1, 1, 1]})
    aa = dask.delayed(a)
    bb = dask.delayed(b)
    df = dd.from_delayed([aa, bb], meta=a)
    assert not df.known_divisions
    df2 = df.set_index('y', sorted=True)
    assert df2.divisions == (0, 1, 1)