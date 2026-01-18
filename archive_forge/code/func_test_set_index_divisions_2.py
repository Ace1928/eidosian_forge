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
def test_set_index_divisions_2():
    df = pd.DataFrame({'x': [1, 2, 3, 4, 5, 6], 'y': list('abdabd')})
    ddf = dd.from_pandas(df, 2)
    result = ddf.set_index('y', divisions=['a', 'c', 'd'])
    assert result.divisions == ('a', 'c', 'd')
    assert list(result.compute(scheduler='sync').index[-2:]) == ['d', 'd']