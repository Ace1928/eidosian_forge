from __future__ import annotations
from operator import add, mul
import pytest
from dask.callbacks import Callback
from dask.diagnostics import ProgressBar
from dask.diagnostics.progress import format_time
from dask.local import get_sync
from dask.threaded import get as get_threaded
def test_with_cache(capsys):
    cachey = pytest.importorskip('cachey')
    from dask.cache import Cache
    c = cachey.Cache(10000)
    cc = Cache(c)
    with cc:
        with ProgressBar():
            assert get_threaded({'x': (mul, 1, 2)}, 'x') == 2
    check_bar_completed(capsys)
    assert c.data['x'] == 2
    with cc:
        with ProgressBar():
            assert get_threaded({'x': (mul, 1, 2), 'y': (mul, 'x', 3)}, 'y') == 6
    check_bar_completed(capsys)