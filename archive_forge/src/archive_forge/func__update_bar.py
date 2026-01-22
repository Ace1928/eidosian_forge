from __future__ import annotations
import contextlib
import sys
import threading
import time
from timeit import default_timer
from dask.callbacks import Callback
from dask.utils import _deprecated
def _update_bar(self, elapsed):
    s = self._state
    if not s:
        self._draw_bar(0, elapsed)
        return
    ndone = len(s['finished'])
    ntasks = sum((len(s[k]) for k in ['ready', 'waiting', 'running'])) + ndone
    if ndone < ntasks:
        self._draw_bar(ndone / ntasks if ntasks else 0, elapsed)