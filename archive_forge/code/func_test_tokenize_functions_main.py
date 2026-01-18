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
def test_tokenize_functions_main():
    script = "\n    def inc(x):\n        return x + 1\n\n    inc2 = inc\n    def sum(x, y):\n        return x + y\n\n    from dask.base import tokenize\n    assert tokenize(inc) != tokenize(sum)\n    # That this is an alias shouldn't matter\n    assert tokenize(inc) == tokenize(inc2)\n\n    def inc(x):\n        return x + 1\n\n    assert tokenize(inc2) != tokenize(inc)\n\n    def inc(y):\n        return y + 1\n\n    assert tokenize(inc2) != tokenize(inc)\n\n    def inc(x):\n        y = x\n        return y + 1\n\n    assert tokenize(inc2) != tokenize(inc)\n    "
    proc = subprocess.run([sys.executable, '-c', textwrap.dedent(script)])
    proc.check_returncode()