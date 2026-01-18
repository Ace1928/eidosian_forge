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
def test_set_index_overlap_2():
    df = pd.DataFrame(index=pd.Index(['A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'C'], name='index'))
    ddf = dd.from_pandas(df, npartitions=2)
    result = ddf.reset_index().repartition(npartitions=8).set_index('index', sorted=True)
    expected = df.reset_index().set_index('index')
    assert_eq(result, expected)
    assert result.npartitions == 8