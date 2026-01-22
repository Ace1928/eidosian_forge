import math
import numbers
import os
import cupy
from ._util import _get_inttype
def _unpack_int2(img, make_copy=False, int_dtype=cupy.int16):
    temp = img.view(int_dtype).reshape(img.shape + (2,))
    if make_copy:
        temp = temp.copy()
    return temp