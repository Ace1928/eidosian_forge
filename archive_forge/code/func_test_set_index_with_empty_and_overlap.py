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
def test_set_index_with_empty_and_overlap():
    df = pd.DataFrame(index=list(range(8)), data={'a': [1, 2, 2, 3, 3, 3, 4, 5], 'b': [1, 1, 0, 0, 0, 1, 0, 0]})
    ddf = dd.from_pandas(df, 4)
    result = ddf[ddf.b == 1].set_index('a', sorted=True)
    expected = df[df.b == 1].set_index('a')
    assert result.divisions == (1.0, 3.0, 3.0)
    assert_eq(result, expected)