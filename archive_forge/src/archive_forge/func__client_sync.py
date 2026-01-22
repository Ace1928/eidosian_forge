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
def _client_sync(self, func: Callable, **kwargs: Any) -> Any:
    """Get the correct client, when method is invoked inside a worker we
        should use `worker_client' instead of default client.

        """
    if self._client is None:
        asynchronous = getattr(self, '_asynchronous', False)
        try:
            distributed.get_worker()
            in_worker = True
        except ValueError:
            in_worker = False
        if in_worker:
            with distributed.worker_client() as client:
                with _set_worker_client(self, client) as this:
                    ret = this.client.sync(func, **kwargs, asynchronous=asynchronous)
                    return ret
                return ret
    return self.client.sync(func, **kwargs, asynchronous=self.client.asynchronous)