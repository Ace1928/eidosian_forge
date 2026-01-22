from __future__ import annotations
import multiprocessing
import threading
import uuid
import weakref
from collections.abc import Hashable, MutableMapping
from typing import Any, ClassVar
from weakref import WeakValueDictionary
def _get_lock_maker(scheduler=None):
    """Returns an appropriate function for creating resource locks.

    Parameters
    ----------
    scheduler : str or None
        Dask scheduler being used.

    See Also
    --------
    dask.utils.get_scheduler_lock
    """
    if scheduler is None:
        return _get_threaded_lock
    elif scheduler == 'threaded':
        return _get_threaded_lock
    elif scheduler == 'multiprocessing':
        return _get_multiprocessing_lock
    elif scheduler == 'distributed':
        try:
            from dask.distributed import Lock as DistributedLock
        except ImportError:
            DistributedLock = None
        return DistributedLock
    else:
        raise KeyError(scheduler)