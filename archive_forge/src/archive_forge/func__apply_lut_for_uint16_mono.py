import numpy
from .Qt import QtGui
from . import functions
from .util.cupy_helper import getCupy
from .util.numba_helper import getNumbaFunctions
def _apply_lut_for_uint16_mono(xp, image, lut):
    augmented_alpha = False
    if not image.flags.c_contiguous:
        image = lut.take(image, axis=0)
        if image.ndim == 3 and image.shape[-1] == 1:
            image = image[..., 0]
        return (image, augmented_alpha)
    lut, augmented_alpha = _convert_2dlut_to_1dlut(xp, lut)
    fn_numba = getNumbaFunctions()
    if xp == numpy and fn_numba is not None:
        image = fn_numba.numba_take(lut, image)
    else:
        image = lut[image]
    if image.dtype == xp.uint32:
        image = image[..., xp.newaxis].view(xp.uint8)
    return (image, augmented_alpha)