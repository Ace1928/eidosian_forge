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
def test_tokenize_slotted_no_value():

    class Foo:
        __slots__ = ('x', 'y')

        def __init__(self, x=None, y=None):
            if x is not None:
                self.x = x
            if y is not None:
                self.y = y
    assert check_tokenize(Foo(x=1)) != check_tokenize(Foo(y=1))
    check_tokenize(Foo())