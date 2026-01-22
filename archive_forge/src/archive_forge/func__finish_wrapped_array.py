import os
import sys
import ctypes
import threading
import logging
import numpy
from ..core import (
def _finish_wrapped_array(self, array):
    """Hardcore way to inject numpy array in bitmap."""
    with self._fi as lib:
        pitch = lib.FreeImage_GetPitch(self._bitmap)
        bits = lib.FreeImage_GetBits(self._bitmap)
        bpp = lib.FreeImage_GetBPP(self._bitmap)
    nchannels = bpp // 8 // array.itemsize
    realwidth = pitch // nchannels
    extra = realwidth - array.shape[-2]
    assert 0 <= extra < 10
    newshape = (array.shape[-1], realwidth, nchannels)
    array2 = numpy.zeros(newshape, array.dtype)
    if nchannels == 1:
        array2[:, :array.shape[-2], 0] = array.T
    else:
        for i in range(nchannels):
            array2[:, :array.shape[-2], i] = array[i, :, :].T
    data_ptr = array2.__array_interface__['data'][0]
    ctypes.memmove(bits, data_ptr, array2.nbytes)
    del array2