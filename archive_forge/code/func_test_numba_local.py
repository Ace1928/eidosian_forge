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
@pytest.mark.skipif('not numba')
def test_numba_local():

    @numba.jit(nopython=True)
    def local_jit(x, y):
        return x + y

    @numba.jit('f8(f8, f8)', nopython=True)
    def local_jit_with_signature(x, y):
        return x + y

    @numba.vectorize(nopython=True)
    def local_vectorize(x, y):
        return x + y

    @numba.vectorize('f8(f8, f8)', nopython=True)
    def local_vectorize_with_signature(x, y):
        return x + y

    @numba.guvectorize(['f8,f8,f8[:]'], '(),()->()')
    def local_guvectorize(x, y, out):
        out[0] = x + y
    all_funcs = [local_jit, local_jit_with_signature, local_vectorize, local_vectorize_with_signature, local_guvectorize]
    tokens = [check_tokenize(func) for func in all_funcs]
    assert len(tokens) == len(set(tokens))