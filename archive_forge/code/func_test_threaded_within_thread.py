from __future__ import annotations
import signal
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.pool import ThreadPool
from time import sleep, time
import pytest
import dask
from dask.system import CPU_COUNT
from dask.threaded import get
from dask.utils_test import add, inc
def test_threaded_within_thread():
    L = []

    def f(i):
        result = get({'x': (lambda: i,)}, 'x', num_workers=2)
        L.append(result)
    before = threading.active_count()
    for _ in range(20):
        t = threading.Thread(target=f, args=(1,))
        t.daemon = True
        t.start()
        t.join()
        assert L == [1]
        del L[:]
    start = time()
    while threading.active_count() > before + 10:
        sleep(0.01)
        assert time() < start + 5