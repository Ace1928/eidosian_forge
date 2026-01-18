from __future__ import annotations
from dask.callbacks import Callback
from dask.local import get_sync
from dask.threaded import get as get_threaded
from dask.utils_test import add
def test_start_callback():
    flag = [False]

    class MyCallback(Callback):

        def _start(self, dsk):
            flag[0] = True
    with MyCallback():
        get_sync({'x': 1}, 'x')
    assert flag[0] is True