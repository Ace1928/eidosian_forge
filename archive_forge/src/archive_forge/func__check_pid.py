from collections import namedtuple
from multiprocessing import current_process
import ctypes
import struct
import numbers
import numpy as np
from .base import _LIB
from .base import RecordIOHandle
from .base import check_call
from .base import c_str
def _check_pid(self, allow_reset=False):
    """Check process id to ensure integrity, reset if in new process."""
    if not self.pid == current_process().pid:
        if allow_reset:
            self.reset()
        else:
            raise RuntimeError('Forbidden operation in multiple processes')