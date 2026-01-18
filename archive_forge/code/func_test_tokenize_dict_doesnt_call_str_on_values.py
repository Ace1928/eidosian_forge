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
def test_tokenize_dict_doesnt_call_str_on_values():

    class C:

        def __dask_tokenize__(self):
            return 'C'

        def __repr__(self):
            assert False
    check_tokenize({1: C(), '2': C()})