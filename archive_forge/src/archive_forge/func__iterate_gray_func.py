import warnings
import numpy as np
from scipy import ndimage as ndi
from .footprints import _footprint_is_sequence, mirror_footprint, pad_footprint
from .misc import default_footprint
from .._shared.utils import DEPRECATED
def _iterate_gray_func(gray_func, image, footprints, out, mode, cval):
    """Helper to call `gray_func` for each footprint in a sequence.

    `gray_func` is a morphology function that accepts `footprint`, `output`,
    `mode` and `cval` keyword arguments (e.g. `scipy.ndimage.grey_erosion`).
    """
    fp, num_iter = footprints[0]
    gray_func(image, footprint=fp, output=out, mode=mode, cval=cval)
    for _ in range(1, num_iter):
        gray_func(out.copy(), footprint=fp, output=out, mode=mode, cval=cval)
    for fp, num_iter in footprints[1:]:
        for _ in range(num_iter):
            gray_func(out.copy(), footprint=fp, output=out, mode=mode, cval=cval)
    return out