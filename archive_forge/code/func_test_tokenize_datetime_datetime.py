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
def test_tokenize_datetime_datetime():
    required = [1, 2, 3]
    optional = [4, 5, 6, 7, datetime.timezone.utc]
    for i in range(len(optional) + 1):
        args = required + optional[:i]
        check_tokenize(datetime.datetime(*args))
    assert check_tokenize(datetime.datetime(1, 2, 3, 4, 5, 6, 7, datetime.timezone.utc)) != check_tokenize(datetime.datetime(2, 2, 3, 4, 5, 6, 7, datetime.timezone.utc))
    assert check_tokenize(datetime.datetime(1, 2, 3, 4, 5, 6, 7, datetime.timezone.utc)) != check_tokenize(datetime.datetime(1, 1, 3, 4, 5, 6, 7, datetime.timezone.utc))
    assert check_tokenize(datetime.datetime(1, 2, 3, 4, 5, 6, 7, datetime.timezone.utc)) != check_tokenize(datetime.datetime(1, 2, 2, 4, 5, 6, 7, datetime.timezone.utc))
    assert check_tokenize(datetime.datetime(1, 2, 3, 4, 5, 6, 7, datetime.timezone.utc)) != check_tokenize(datetime.datetime(1, 2, 3, 3, 5, 6, 7, datetime.timezone.utc))
    assert check_tokenize(datetime.datetime(1, 2, 3, 4, 5, 6, 7, datetime.timezone.utc)) != check_tokenize(datetime.datetime(1, 2, 3, 4, 4, 6, 7, datetime.timezone.utc))
    assert check_tokenize(datetime.datetime(1, 2, 3, 4, 5, 6, 7, datetime.timezone.utc)) != check_tokenize(datetime.datetime(1, 2, 3, 4, 5, 5, 7, datetime.timezone.utc))
    assert check_tokenize(datetime.datetime(1, 2, 3, 4, 5, 6, 7, datetime.timezone.utc)) != check_tokenize(datetime.datetime(1, 2, 3, 4, 5, 6, 6, datetime.timezone.utc))
    assert check_tokenize(datetime.datetime(1, 2, 3, 4, 5, 6, 7, datetime.timezone.utc)) != check_tokenize(datetime.datetime(1, 2, 3, 4, 5, 6, 7, None))