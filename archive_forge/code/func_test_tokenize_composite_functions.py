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
def test_tokenize_composite_functions():
    assert check_tokenize(partial(f2, b=2)) != check_tokenize(partial(f2, b=3))
    assert check_tokenize(partial(f1, b=2)) != check_tokenize(partial(f2, b=2))
    assert check_tokenize(compose(f2, f3)) != check_tokenize(compose(f2, f1))
    assert check_tokenize(curry(f2)) != check_tokenize(curry(f1))
    assert check_tokenize(curry(f2, b=1)) != check_tokenize(curry(f2, b=2))