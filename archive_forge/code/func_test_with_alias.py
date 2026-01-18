from __future__ import annotations
from operator import add, mul
import pytest
from dask.callbacks import Callback
from dask.diagnostics import ProgressBar
from dask.diagnostics.progress import format_time
from dask.local import get_sync
from dask.threaded import get as get_threaded
def test_with_alias(capsys):
    dsk = {'a': 1, 'b': 2, 'c': (add, 'a', 'b'), 'd': (add, 1, 2), 'e': 'd', 'f': (mul, 'e', 'c')}
    with ProgressBar():
        get_threaded(dsk, 'f')
    check_bar_completed(capsys)