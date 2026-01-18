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
def test_tokenize_circular_recursion():
    a = [1, 2]
    a[0] = a
    b = [1, 3]
    b[0] = b
    assert check_tokenize(a) != check_tokenize(b)
    c = [[], []]
    c[0].append(c[0])
    c[1].append(c[1])
    d = [[], []]
    d[0].append(d[1])
    d[1].append(d[0])
    assert check_tokenize(c) != check_tokenize(d)
    e = {}
    e[0] = e
    check_tokenize(e)