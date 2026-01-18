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
def test_tokenize_dataclass():
    a1 = ADataClass(1)
    a2 = ADataClass(2)
    check_tokenize(a1)
    assert check_tokenize(a1) != check_tokenize(a2)
    b1 = BDataClass(1)
    assert check_tokenize(ADataClass) != check_tokenize(BDataClass)
    assert check_tokenize(a1) != check_tokenize(b1)

    class SubA(ADataClass):
        pass
    assert dataclasses.is_dataclass(SubA)
    assert check_tokenize(ADataClass) != check_tokenize(SubA)
    assert check_tokenize(SubA(1)) != check_tokenize(a1)
    ADataClassRedefinedDifferently = dataclasses.make_dataclass('ADataClass', [('a', Union[int, str])])
    assert check_tokenize(a1) != check_tokenize(ADataClassRedefinedDifferently(1))
    nv = NoValueDataClass()
    check_tokenize(nv)