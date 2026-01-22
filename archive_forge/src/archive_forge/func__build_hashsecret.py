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
def _build_hashsecret():
    """Read hash secret from the Python process

    Returns
    -------
    info : dict
        - keys are "djbx33a_suffix", "siphash_k0", siphash_k1".
        - values are the namedtuple[symbol:str, value:int]
    """
    pyhashsecret = _Py_HashSecret_t.in_dll(pythonapi, '_Py_HashSecret')
    info = {}

    def inject(name, val):
        symbol_name = '_numba_hashsecret_{}'.format(name)
        val = ctypes.c_uint64(val)
        addr = ctypes.addressof(val)
        ll.add_symbol(symbol_name, addr)
        info[name] = _hashsecret_entry(symbol=symbol_name, value=val)
    inject('djbx33a_suffix', pyhashsecret.djbx33a.suffix)
    inject('siphash_k0', pyhashsecret.siphash.k0)
    inject('siphash_k1', pyhashsecret.siphash.k1)
    return info