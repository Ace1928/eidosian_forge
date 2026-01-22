from contextlib import contextmanager
import sys
import threading
import traceback
from numba.core import types
import numpy as np
from numba.np import numpy_support
from .vector_types import vector_types
class GridGroup:
    """
    Used to implement the grid group.
    """

    def sync(self):
        threading.current_thread().syncthreads()