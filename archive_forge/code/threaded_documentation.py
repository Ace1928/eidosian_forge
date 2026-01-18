from __future__ import annotations
import atexit
import multiprocessing.pool
import sys
import threading
from collections import defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import Executor, ThreadPoolExecutor
from threading import Lock, current_thread
from dask import config
from dask.local import MultiprocessingPoolExecutor, get_async
from dask.system import CPU_COUNT
from dask.typing import Key
Threaded cached implementation of dask.get

    Parameters
    ----------

    dsk: dict
        A dask dictionary specifying a workflow
    keys: key or list of keys
        Keys corresponding to desired data
    num_workers: integer of thread count
        The number of threads to use in the ThreadPool that will actually execute tasks
    cache: dict-like (optional)
        Temporary storage of results

    Examples
    --------
    >>> inc = lambda x: x + 1
    >>> add = lambda x, y: x + y
    >>> dsk = {'x': 1, 'y': 2, 'z': (inc, 'x'), 'w': (add, 'z', 'y')}
    >>> get(dsk, 'w')
    4
    >>> get(dsk, ['w', 'y'])
    (4, 2)
    