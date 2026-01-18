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
def test_tokenize_scalar():
    assert check_tokenize(np.int64(1)) != check_tokenize(1)
    assert check_tokenize(np.int64(1)) != check_tokenize(np.int32(1))
    assert check_tokenize(np.int64(1)) != check_tokenize(np.uint32(1))
    assert check_tokenize(np.int64(1)) != check_tokenize('1')
    assert check_tokenize(np.int64(1)) != check_tokenize(np.float64(1))