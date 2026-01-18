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
@pytest.mark.skipif(DASK_EXPR_ENABLED, reason='not available')
def test_set_index_divisions_compute():
    deprecated = pytest.warns(FutureWarning, match="the 'compute' keyword is deprecated")
    with deprecated:
        d2 = d.set_index('b', divisions=[0, 2, 9], compute=False)
    with deprecated:
        d3 = d.set_index('b', divisions=[0, 2, 9], compute=True)
    assert_eq(d2, d3)
    assert_eq(d2, full.set_index('b'))
    assert_eq(d3, full.set_index('b'))
    assert len(d2.dask) > len(d3.dask)
    with deprecated:
        d4 = d.set_index(d.b, divisions=[0, 2, 9], compute=False)
    with deprecated:
        d5 = d.set_index(d.b, divisions=[0, 2, 9], compute=True)
    exp = full.copy()
    exp.index = exp.b
    assert_eq(d4, d5)
    assert_eq(d4, exp)
    assert_eq(d5, exp)
    assert len(d4.dask) > len(d5.dask)