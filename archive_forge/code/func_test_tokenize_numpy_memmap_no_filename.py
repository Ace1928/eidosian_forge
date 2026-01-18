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
def test_tokenize_numpy_memmap_no_filename():
    with tmpfile('.npy') as fn1, tmpfile('.npy') as fn2:
        x = np.arange(5)
        np.save(fn1, x)
        np.save(fn2, x)
        a = np.load(fn1, mmap_mode='r')
        b = a + a
        assert check_tokenize(b) == check_tokenize(b)