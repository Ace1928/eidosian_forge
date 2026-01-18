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
@pytest.mark.skipif('not pd')
def test_tokenize_pandas():
    a = pd.DataFrame({'x': [1, 2, 3], 'y': ['4', 'asd', None]}, index=[1, 2, 3])
    b = pd.DataFrame({'x': [1, 2, 3], 'y': ['4', 'asd', None]}, index=[1, 2, 3])
    assert check_tokenize(a) == check_tokenize(b)
    b.index.name = 'foo'
    assert check_tokenize(a) != check_tokenize(b)
    a = pd.DataFrame({'x': [1, 2, 3], 'y': ['a', 'b', 'a']})
    b = pd.DataFrame({'x': [1, 2, 3], 'y': ['a', 'b', 'a']})
    a['z'] = a.y.astype('category')
    assert check_tokenize(a) != check_tokenize(b)
    b['z'] = a.y.astype('category')
    assert check_tokenize(a) == check_tokenize(b)