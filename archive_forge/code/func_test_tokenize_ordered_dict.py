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
def test_tokenize_ordered_dict():
    from collections import OrderedDict
    a = OrderedDict([('a', 1), ('b', 2)])
    b = OrderedDict([('a', 1), ('b', 2)])
    c = OrderedDict([('b', 2), ('a', 1)])
    assert check_tokenize(a) == check_tokenize(b)
    assert check_tokenize(a) != check_tokenize(c)