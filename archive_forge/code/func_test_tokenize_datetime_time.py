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
def test_tokenize_datetime_time():
    check_tokenize(datetime.time(1, 2, 3, 4, datetime.timezone.utc))
    check_tokenize(datetime.time(1, 2, 3, 4))
    check_tokenize(datetime.time(1, 2, 3))
    check_tokenize(datetime.time(1, 2))
    assert check_tokenize(datetime.time(1, 2, 3, 4, datetime.timezone.utc)) != check_tokenize(datetime.time(2, 2, 3, 4, datetime.timezone.utc))
    assert check_tokenize(datetime.time(1, 2, 3, 4, datetime.timezone.utc)) != check_tokenize(datetime.time(1, 3, 3, 4, datetime.timezone.utc))
    assert check_tokenize(datetime.time(1, 2, 3, 4, datetime.timezone.utc)) != check_tokenize(datetime.time(1, 2, 4, 4, datetime.timezone.utc))
    assert check_tokenize(datetime.time(1, 2, 3, 4, datetime.timezone.utc)) != check_tokenize(datetime.time(1, 2, 3, 5, datetime.timezone.utc))
    assert check_tokenize(datetime.time(1, 2, 3, 4, datetime.timezone.utc)) != check_tokenize(datetime.time(1, 2, 3, 4))