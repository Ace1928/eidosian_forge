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
@pytest.mark.skipif('not np')
def test_tokenize_numpy_memmap():
    with tmpfile('.npy') as fn:
        x1 = np.arange(5)
        np.save(fn, x1)
        y = check_tokenize(np.load(fn, mmap_mode='r'))
    with tmpfile('.npy') as fn:
        x2 = np.arange(5)
        np.save(fn, x2)
        z = check_tokenize(np.load(fn, mmap_mode='r'))
    assert check_tokenize(x1) == check_tokenize(x2)
    assert y == z
    with tmpfile('.npy') as fn:
        x = np.random.normal(size=(10, 10))
        np.save(fn, x)
        mm = np.load(fn, mmap_mode='r')
        mm2 = np.load(fn, mmap_mode='r')
        a = check_tokenize(mm[0, :])
        b = check_tokenize(mm[1, :])
        c = check_tokenize(mm[0:3, :])
        d = check_tokenize(mm[:, 0])
        assert len({a, b, c, d}) == 4
        assert check_tokenize(mm) == check_tokenize(mm2)
        assert check_tokenize(mm[1, :]) == check_tokenize(mm2[1, :])