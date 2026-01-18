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
def test_tokenize_offset():
    for offset in [pd.offsets.Second(1), pd.offsets.MonthBegin(2), pd.offsets.Day(1), pd.offsets.BQuarterEnd(2), pd.DateOffset(years=1), pd.DateOffset(months=7), pd.DateOffset(days=10)]:
        check_tokenize(offset)