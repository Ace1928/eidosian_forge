from __future__ import annotations
from typing import Any
from unittest import mock
import pytest
from xarray.backends.lru_cache import LRUCache
def test_on_evict_trivial() -> None:
    on_evict = mock.Mock()
    cache = LRUCache(maxsize=0, on_evict=on_evict)
    cache['x'] = 1
    on_evict.assert_called_once_with('x', 1)