import collections
import ctypes
import re
import numpy as np
from numba.core import errors, types
from numba.core.typing.templates import signature
from numba.np import npdatetime_helpers
from numba.core.errors import TypingError
from numba.core.cgutils import is_nonelike   # noqa: F401
def _check_struct_alignment(rec, fields):
    """Check alignment compatibility with Numpy"""
    if rec.aligned:
        for k, dt in zip(fields['names'], fields['formats']):
            llvm_align = rec.alignof(k)
            npy_align = dt.alignment
            if llvm_align is not None and npy_align != llvm_align:
                msg = 'NumPy is using a different alignment ({}) than Numba/LLVM ({}) for {}. This is likely a NumPy bug.'
                raise ValueError(msg.format(npy_align, llvm_align, dt))