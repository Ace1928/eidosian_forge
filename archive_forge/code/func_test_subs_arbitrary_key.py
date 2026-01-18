from __future__ import annotations
import pickle
from collections import namedtuple
import pytest
from dask.core import (
from dask.utils_test import GetFunctionTestMixin, add, inc
def test_subs_arbitrary_key():
    key = (1.2, 'foo', (3,))
    assert subs((id, key), key, 1) == (id, 1)