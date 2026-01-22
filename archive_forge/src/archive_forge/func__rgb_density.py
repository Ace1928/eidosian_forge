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
@nb.jit(nopython=True, nogil=True, cache=True)
def _rgb_density(arr, px=1):
    """Compute a density heuristic of an image.

    The density is a number in [0, 1], and indicates the normalized mean number
    of non-empty pixels that have neighbors in the given px radius.
    """
    M, N = arr.shape
    cnt = has_neighbors = 0
    for y in range(0, M):
        for x in range(0, N):
            if arr[y, x] >> 24 & 255:
                cnt += 1
                neighbors = 0
                for i in range(max(0, y - px), min(y + px + 1, M)):
                    for j in range(max(0, x - px), min(x + px + 1, N)):
                        if arr[i, j] >> 24 & 255:
                            neighbors += 1
                if neighbors > 1:
                    has_neighbors += 1
    return has_neighbors / cnt if cnt else np.inf