import enum
import functools
import random
import threading
import time
import grpc
from tensorboard import version
from tensorboard.util import tb_logging
def _set_active_future(self, grpc_future):
    if grpc_future is None:
        raise RuntimeError('_set_active_future invoked with grpc_future=None.')
    with self._active_grpc_future_lock:
        self._active_grpc_future = grpc_future