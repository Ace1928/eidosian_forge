import pytest
import types
import sys
import collections.abc
from functools import wraps
import gc
from .conftest import mock_sleep
from .. import (
from .. import _impl
@pytest.mark.skipif(not hasattr(sys, 'getrefcount'), reason='CPython only')
def test_refcnt():
    x = object()
    print(sys.getrefcount(x))
    print(sys.getrefcount(x))
    print(sys.getrefcount(x))
    print(sys.getrefcount(x))
    base_count = sys.getrefcount(x)
    l = [_impl._wrap(x) for _ in range(100)]
    print(sys.getrefcount(x))
    print(sys.getrefcount(x))
    print(sys.getrefcount(x))
    assert sys.getrefcount(x) >= base_count + 100
    l2 = [_impl._unwrap(box) for box in l]
    assert sys.getrefcount(x) >= base_count + 200
    print(sys.getrefcount(x))
    print(sys.getrefcount(x))
    print(sys.getrefcount(x))
    print(sys.getrefcount(x))
    del l
    print(sys.getrefcount(x))
    print(sys.getrefcount(x))
    print(sys.getrefcount(x))
    del l2
    print(sys.getrefcount(x))
    print(sys.getrefcount(x))
    print(sys.getrefcount(x))
    assert sys.getrefcount(x) == base_count
    print(sys.getrefcount(x))