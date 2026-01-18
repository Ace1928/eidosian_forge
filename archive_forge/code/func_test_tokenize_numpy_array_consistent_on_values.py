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
@pytest.mark.skipif('not np')
def test_tokenize_numpy_array_consistent_on_values():
    assert check_tokenize(np.random.RandomState(1234).random_sample(1000)) == check_tokenize(np.random.RandomState(1234).random_sample(1000))