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
@pytest.mark.parametrize('module', ['random', pytest.param('np.random', marks=pytest.mark.skipif('not np'))])
def test_tokenize_random_functions(module):
    """random.random() and other methods of the global random state do not compare as
    equal to themselves after a pickle roundtrip"""
    module = eval(module)
    a = module.random
    b = pickle.loads(pickle.dumps(a))
    assert check_tokenize(a) == check_tokenize(b)
    a()
    c = pickle.loads(pickle.dumps(a))
    assert check_tokenize(a) == check_tokenize(c)
    assert check_tokenize(a) != check_tokenize(b)
    module.seed(123)
    d = pickle.loads(pickle.dumps(a))
    assert check_tokenize(a) == check_tokenize(d)
    assert check_tokenize(a) != check_tokenize(c)