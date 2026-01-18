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
def test_tokenize_callable_class():

    class C:

        def __init__(self, x):
            self.x = x

        def __call__(self):
            return self.x

    class D(C):
        pass
    a, b, c = (C(1), C(2), D(1))
    assert check_tokenize(a) != check_tokenize(b)
    assert check_tokenize(a) != check_tokenize(c)