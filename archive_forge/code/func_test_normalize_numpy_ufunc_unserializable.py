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
def test_normalize_numpy_ufunc_unserializable():
    inc = np.frompyfunc(lambda x: x + 1, 1, 1)
    with dask.config.set({'tokenize.ensure-deterministic': False}):
        assert tokenize(inc) != tokenize(inc)
        assert normalize_token(inc) != normalize_token(inc)
    with dask.config.set({'tokenize.ensure-deterministic': True}):
        with pytest.raises(TokenizationError, match='Cannot tokenize.*dask\\.array\\.ufunc.*instead'):
            tokenize(inc)
    assert tokenize(inc, ensure_deterministic=False) != tokenize(inc, ensure_deterministic=False)
    with pytest.raises(TokenizationError, match='Cannot tokenize'):
        tokenize(inc, ensure_deterministic=True)