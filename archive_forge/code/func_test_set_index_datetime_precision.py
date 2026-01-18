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
@pytest.mark.parametrize('unit', ['ns', 'us'])
def test_set_index_datetime_precision(unit):
    df = pd.DataFrame([[1567703791155681, 1], [1567703792155681, 2], [1567703790155681, 0], [1567703793155681, 3]], columns=['ts', 'rank'])
    df.ts = pd.to_datetime(df.ts, unit=unit)
    ddf = dd.from_pandas(df, npartitions=2)
    ddf = ddf.set_index('ts')
    assert_eq(ddf, df.set_index('ts'))