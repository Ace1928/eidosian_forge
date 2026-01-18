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
def test_tokenize_method():

    class Foo:

        def __init__(self, x):
            self.x = x

        def __dask_tokenize__(self):
            return self.x

        def hello(self):
            return 'Hello world'
    a, b = (Foo(1), Foo(2))
    assert check_tokenize(a) != check_tokenize(b)
    assert check_tokenize(a.hello) != check_tokenize(b.hello)
    before = check_tokenize(a)
    normalize_token.register(Foo, lambda self: self.x + 1)
    after = check_tokenize(a)
    assert before != after
    del normalize_token._lookup[Foo]