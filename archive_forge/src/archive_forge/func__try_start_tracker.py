import collections
import logging
import platform
import socket
import warnings
from collections import defaultdict
from contextlib import contextmanager
from functools import partial, update_wrapper
from threading import Thread
from typing import (
import numpy
from . import collective, config
from ._typing import _T, FeatureNames, FeatureTypes, ModelIn
from .callback import TrainingCallback
from .compat import DataFrame, LazyLoader, concat, lazy_isinstance
from .core import (
from .data import _is_cudf_ser, _is_cupy_array
from .sklearn import (
from .tracker import RabitTracker, get_host_ip
from .training import train as worker_train
def _try_start_tracker(n_workers: int, addrs: List[Union[Optional[str], Optional[Tuple[str, int]]]]) -> Dict[str, Union[int, str]]:
    env: Dict[str, Union[int, str]] = {'DMLC_NUM_WORKER': n_workers}
    try:
        if isinstance(addrs[0], tuple):
            host_ip = addrs[0][0]
            port = addrs[0][1]
            rabit_tracker = RabitTracker(host_ip=get_host_ip(host_ip), n_workers=n_workers, port=port, use_logger=False)
        else:
            addr = addrs[0]
            assert isinstance(addr, str) or addr is None
            host_ip = get_host_ip(addr)
            rabit_tracker = RabitTracker(host_ip=host_ip, n_workers=n_workers, use_logger=False, sortby='task')
        env.update(rabit_tracker.worker_envs())
        rabit_tracker.start(n_workers)
        thread = Thread(target=rabit_tracker.join)
        thread.daemon = True
        thread.start()
    except socket.error as e:
        if len(addrs) < 2 or e.errno != 99:
            raise
        LOGGER.warning("Failed to bind address '%s', trying to use '%s' instead.", str(addrs[0]), str(addrs[1]))
        env = _try_start_tracker(n_workers, addrs[1:])
    return env