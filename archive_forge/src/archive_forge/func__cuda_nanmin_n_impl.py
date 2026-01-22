from __future__ import annotations
from math import ceil, isnan
from packaging.version import Version
import numba
import numpy as np
from numba import cuda
@cuda.jit(device=True)
def _cuda_nanmin_n_impl(ret_pixel, other_pixel):
    """Single pixel implementation of nanmin_n_in_place.
    ret_pixel and other_pixel are both 1D arrays of the same length.

    Walk along other_pixel a value at a time, find insertion index in
    ret_pixel and shift values along to insert.  Next other_pixel value is
    inserted at a higher index, so this walks the two pixel arrays just once
    each.
    """
    n = len(ret_pixel)
    istart = 0
    for other_value in other_pixel:
        if isnan(other_value):
            break
        else:
            for i in range(istart, n):
                if isnan(ret_pixel[i]) or other_value < ret_pixel[i]:
                    istart = cuda_shift_and_insert(ret_pixel, other_value, i)
                    break