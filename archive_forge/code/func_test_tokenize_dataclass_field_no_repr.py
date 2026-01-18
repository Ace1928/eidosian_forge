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
def test_tokenize_dataclass_field_no_repr():
    A = dataclasses.make_dataclass('A', [('param', float, dataclasses.field(repr=False))], namespace={'__dask_tokenize__': lambda self: self.param})
    a1, a2 = (A(1), A(2))
    assert check_tokenize(a1) != check_tokenize(a2)