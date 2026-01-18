from __future__ import annotations
import pytest
import dask
from dask.local import finish_task, get_sync, sortkey, start_state_from_dask
from dask.order import order
from dask.utils_test import GetFunctionTestMixin, add, inc
def test_start_state_with_redirects():
    dsk = {'x': 1, 'y': 'x', 'z': (inc, 'y')}
    result = start_state_from_dask(dsk)
    assert result['cache'] == {'x': 1}