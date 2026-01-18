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
@pytest.mark.parametrize('other', [(1, 10, 2), (5, 15, 2), (5, 10, 1)])
def test_tokenize_range(other):
    assert check_tokenize(range(5, 10, 2)) != check_tokenize(range(*other))