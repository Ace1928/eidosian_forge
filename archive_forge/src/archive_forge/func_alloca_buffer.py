from collections import namedtuple
import contextlib
import pickle
import hashlib
import sys
from llvmlite import ir
from llvmlite.ir import Constant
import ctypes
from numba import _helperlib
from numba.core import (
from numba.core.utils import PYVERSION
def alloca_buffer(self):
    """
        Return a pointer to a stack-allocated, zero-initialized Py_buffer.
        """
    ptr = cgutils.alloca_once_value(self.builder, Constant(self.py_buffer_t, None))
    return ptr