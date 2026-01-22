from numpy.testing import assert_equal, assert_
from pytest import raises as assert_raises
import time
import pytest
import ctypes
import threading
from scipy._lib import _ccallback_c as _test_ccallback_cython
from scipy._lib import _test_ccallback
from scipy._lib._ccallback import LowLevelCallable
def _get_ctypes_data():
    value = ctypes.c_double(2.0)
    return ctypes.cast(ctypes.pointer(value), ctypes.c_voidp)