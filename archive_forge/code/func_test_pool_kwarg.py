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
@pytest.mark.parametrize('pool_typ', [ThreadPool, ThreadPoolExecutor])
def test_pool_kwarg(pool_typ):

    def f():
        sleep(0.01)
        return threading.get_ident()
    dsk = {('x', i): (f,) for i in range(30)}
    dsk['x'] = (len, (set, [('x', i) for i in range(len(dsk))]))
    with pool_typ(3) as pool:
        assert get(dsk, 'x', pool=pool) == 3