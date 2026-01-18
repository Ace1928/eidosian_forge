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
def test_set_index_empty_partition():
    test_vals = [1, 2, 3]
    converters = [int, float, str, lambda x: pd.to_datetime(x, unit='ns')]
    for conv in converters:
        df = pd.DataFrame([{'x': conv(i), 'y': i} for i in test_vals], columns=['x', 'y'])
        ddf = dd.concat([dd.from_pandas(df, npartitions=1), dd.from_pandas(df[df.y > df.y.max()], npartitions=1)])
        assert any((ddf.get_partition(p).compute().empty for p in range(ddf.npartitions)))
        assert assert_eq(ddf.set_index('x'), df.set_index('x'))