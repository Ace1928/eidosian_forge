from numpy.testing import assert_equal, assert_
from pytest import raises as assert_raises
import time
import pytest
import ctypes
import threading
from scipy._lib import _ccallback_c as _test_ccallback_cython
from scipy._lib import _test_ccallback
from scipy._lib._ccallback import LowLevelCallable
def _get_cffi_data():
    if not HAVE_CFFI:
        pytest.skip('cffi not installed')
    ffi = cffi.FFI()
    return ffi.new('double *', 2.0)