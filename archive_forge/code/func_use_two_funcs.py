import sys
import numpy as np
import numba.core.typing.cffi_utils as cffi_support
from numba.tests.support import import_dynamic, temp_directory
from numba.core.types import complex128
def use_two_funcs(x):
    return cffi_sin(x) - cffi_cos(x)