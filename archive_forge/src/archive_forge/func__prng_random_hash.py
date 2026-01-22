import math
import numpy as np
import sys
import ctypes
import warnings
from collections import namedtuple
import llvmlite.binding as ll
from llvmlite import ir
from numba import literal_unroll
from numba.core.extending import (
from numba.core import errors
from numba.core import types, utils
from numba.core.unsafe.bytes import grab_byte, grab_uint64_t
from numba.cpython.randomimpl import (const_int, get_next_int, get_next_int32,
from ctypes import (  # noqa
@intrinsic
def _prng_random_hash(tyctx):

    def impl(cgctx, builder, signature, args):
        state_ptr = get_state_ptr(cgctx, builder, 'internal')
        bits = const_int(_hash_width)
        if _hash_width == 32:
            value = get_next_int32(cgctx, builder, state_ptr)
        else:
            value = get_next_int(cgctx, builder, state_ptr, bits, False)
        return value
    sig = _Py_hash_t()
    return (sig, impl)