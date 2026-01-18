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
def test_shuffle_values_raises():
    df = pd.DataFrame({'a': [1, 3, 2]})
    ddf = dd.from_pandas(df, npartitions=3)
    with pytest.raises(ValueError, match="na_position must be either 'first' or 'last'"):
        ddf.sort_values(by='a', na_position='invalid')