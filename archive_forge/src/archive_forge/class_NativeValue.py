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
class NativeValue(object):
    """
    Encapsulate the result of converting a Python object to a native value,
    recording whether the conversion was successful and how to cleanup.
    """

    def __init__(self, value, is_error=None, cleanup=None):
        self.value = value
        self.is_error = is_error if is_error is not None else cgutils.false_bit
        self.cleanup = cleanup