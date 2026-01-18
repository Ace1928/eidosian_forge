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
@pytest.mark.skipif('not pd')
def test_tokenize_pandas_mixed_unicode_bytes():
    df = pd.DataFrame({'ö'.encode(): [1, 2, 3], 'ö': ['ö', 'ö'.encode(), None]}, index=[1, 2, 3])
    check_tokenize(df)