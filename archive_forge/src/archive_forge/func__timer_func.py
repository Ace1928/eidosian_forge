from __future__ import annotations
import contextlib
import sys
import threading
import time
from timeit import default_timer
from dask.callbacks import Callback
from dask.utils import _deprecated
def _timer_func(self):
    """Background thread for updating the progress bar"""
    while self._running:
        elapsed = default_timer() - self._start_time
        if elapsed > self._minimum:
            self._update_bar(elapsed)
        time.sleep(self._dt)