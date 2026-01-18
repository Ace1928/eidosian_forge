from __future__ import annotations
import dataclasses
import datetime
import decimal
import operator
import pathlib
import pickle
import random
import subprocess
import sys
import textwrap
from enum import Enum, Flag, IntEnum, IntFlag
from typing import Union
import cloudpickle
import pytest
from tlz import compose, curry, partial
import dask
from dask.base import TokenizationError, normalize_token, tokenize
from dask.core import literal
from dask.utils import tmpfile
from dask.utils_test import import_or_none
@pytest.mark.skipif('not sp')
@pytest.mark.parametrize('cls_name', ('dok',))
def test_tokenize_dense_sparse_array(cls_name):
    rng = np.random.RandomState(1234)
    a = sp.rand(10, 100, random_state=rng).asformat(cls_name)
    b = a.copy()
    assert check_tokenize(a) == check_tokenize(b)
    if hasattr(b, 'data'):
        b.data[:10] = 1
    elif cls_name == 'dok':
        b[3, 3] = 1
    else:
        raise ValueError
    check_tokenize(b)
    assert check_tokenize(a) != check_tokenize(b)
    b = a.copy().asformat('coo')
    b.row[:10] = np.arange(10)
    b = b.asformat(cls_name)
    assert check_tokenize(a) != check_tokenize(b)