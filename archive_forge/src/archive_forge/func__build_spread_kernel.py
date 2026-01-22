from __future__ import annotations
from collections.abc import Iterator
from io import BytesIO
import warnings
import numpy as np
import numba as nb
import toolz as tz
import xarray as xr
import dask.array as da
from PIL.Image import fromarray
from datashader.colors import rgb, Sets1to3
from datashader.utils import nansum_missing, ngjit
@tz.memoize
def _build_spread_kernel(how, is_image):
    """Build a spreading kernel for a given composite operator"""
    from datashader.composite import composite_op_lookup, validate_operator
    validate_operator(how, is_image=True)
    op = composite_op_lookup[how + ('' if is_image else '_arr')]

    @ngjit
    def kernel(arr, mask, out):
        M, N = arr.shape
        w = mask.shape[0]
        for y in range(M):
            for x in range(N):
                el = arr[y, x]
                process_image = is_image and int(el) >> 24 & 255
                process_array = not is_image and (not np.isnan(el))
                if process_image or process_array:
                    for i in range(w):
                        for j in range(w):
                            if mask[i, j]:
                                if el == 0:
                                    result = out[i + y, j + x]
                                if out[i + y, j + x] == 0:
                                    result = el
                                else:
                                    result = op(el, out[i + y, j + x])
                                out[i + y, j + x] = result
    return kernel