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
@pytest.mark.parametrize('drop', [True, False])
def test_set_index_drop(drop):
    pdf = pd.DataFrame({'A': list('ABAABBABAA'), 'B': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'C': [1, 2, 3, 2, 1, 3, 2, 4, 2, 3]})
    ddf = dd.from_pandas(pdf, 3)
    assert_eq(ddf.set_index('A', drop=drop), pdf.set_index('A', drop=drop))
    assert_eq(ddf.set_index('B', drop=drop), pdf.set_index('B', drop=drop))
    assert_eq(ddf.set_index('C', drop=drop), pdf.set_index('C', drop=drop))
    assert_eq(ddf.set_index(ddf.A, drop=drop), pdf.set_index(pdf.A, drop=drop))
    assert_eq(ddf.set_index(ddf.B, drop=drop), pdf.set_index(pdf.B, drop=drop))
    assert_eq(ddf.set_index(ddf.C, drop=drop), pdf.set_index(pdf.C, drop=drop))
    pdf = pd.DataFrame({0: list('ABAABBABAA'), 1: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 2: [1, 2, 3, 2, 1, 3, 2, 4, 2, 3]})
    ddf = dd.from_pandas(pdf, 3)
    assert_eq(ddf.set_index(0, drop=drop), pdf.set_index(0, drop=drop))
    assert_eq(ddf.set_index(2, drop=drop), pdf.set_index(2, drop=drop))