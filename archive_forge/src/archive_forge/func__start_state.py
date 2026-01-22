from __future__ import annotations
from dask.callbacks import Callback
from dask.local import get_sync
from dask.threaded import get as get_threaded
from dask.utils_test import add
def _start_state(self, dsk, state):
    flag[0] = True
    assert dsk['x'] == 1
    assert len(state['cache']) == 1