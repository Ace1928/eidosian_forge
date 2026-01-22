from numpy.testing import assert_equal, assert_
from pytest import raises as assert_raises
import time
import pytest
import ctypes
import threading
from scipy._lib import _ccallback_c as _test_ccallback_cython
from scipy._lib import _test_ccallback
from scipy._lib._ccallback import LowLevelCallable
def _get_cffi_func(base, signature):
    if not HAVE_CFFI:
        pytest.skip('cffi not installed')
    voidp = ctypes.cast(base, ctypes.c_void_p)
    address = voidp.value
    ffi = cffi.FFI()
    func = ffi.cast(signature, address)
    return func