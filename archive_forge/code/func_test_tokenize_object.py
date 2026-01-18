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
def test_tokenize_object():
    with dask.config.set({'tokenize.ensure-deterministic': False}):
        o = object()
        assert tokenize(o) == tokenize(o)
        assert tokenize(object()) != tokenize(object())
        assert len({tokenize(object()) for _ in range(100)}) == 100
        assert normalize_token(o) == normalize_token(o)
    with dask.config.set({'tokenize.ensure-deterministic': True}):
        with pytest.raises(TokenizationError, match='deterministic'):
            tokenize(o)
        with pytest.raises(TokenizationError, match='deterministic'):
            normalize_token(o)
    assert tokenize(o, ensure_deterministic=False) == tokenize(o, ensure_deterministic=False)
    with pytest.raises(TokenizationError, match='deterministic'):
        tokenize(o, ensure_deterministic=True)