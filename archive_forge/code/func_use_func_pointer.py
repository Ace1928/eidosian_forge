import sys
import numpy as np
import numba.core.typing.cffi_utils as cffi_support
from numba.tests.support import import_dynamic, temp_directory
from numba.core.types import complex128
def use_func_pointer(fa, fb, x):
    if x > 0:
        return fa(x)
    else:
        return fb(x)