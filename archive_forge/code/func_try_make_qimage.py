import numpy
from .Qt import QtGui
from . import functions
from .util.cupy_helper import getCupy
from .util.numba_helper import getNumbaFunctions
def try_make_qimage(image, *, levels, lut):
    """
    Internal function to make an QImage from an ndarray without going
    through the full generality of makeARGB().
    Only certain combinations of input arguments are supported.
    """
    cp = getCupy()
    xp = cp.get_array_module(image) if cp else numpy
    if image.dtype.kind == 'f' and levels is None:
        return None
    if levels is not None:
        levels = xp.asarray(levels)
        if levels.ndim != 1:
            return None
    if lut is not None and lut.dtype != xp.uint8:
        raise ValueError('lut dtype must be uint8')
    augmented_alpha = False
    if image.dtype.kind == 'f':
        image, levels, lut, augmented_alpha = _rescale_float_mono(xp, image, levels, lut)
    elif image.dtype in (xp.ubyte, xp.uint16):
        image, levels, lut, augmented_alpha = _try_combine_lut(xp, image, levels, lut)
    ubyte_nolvl = image.dtype == xp.ubyte and levels is None
    is_passthru8 = ubyte_nolvl and lut is None
    is_indexed8 = ubyte_nolvl and image.ndim == 2 and (lut is not None) and (lut.shape[0] <= 256)
    is_passthru16 = image.dtype == xp.uint16 and levels is None and (lut is None)
    can_grayscale16 = is_passthru16 and image.ndim == 2 and hasattr(QtGui.QImage.Format, 'Format_Grayscale16')
    is_rgba64 = is_passthru16 and image.ndim == 3 and (image.shape[2] == 4)
    supported = is_passthru8 or is_indexed8 or can_grayscale16 or is_rgba64
    if not supported:
        return None
    if xp == cp:
        image = image.get()
    image = numpy.ascontiguousarray(image)
    fmt = None
    ctbl = None
    if is_passthru8:
        if image.ndim == 2:
            fmt = QtGui.QImage.Format.Format_Grayscale8
        elif image.shape[2] == 3:
            fmt = QtGui.QImage.Format.Format_RGB888
        elif image.shape[2] == 4:
            if augmented_alpha:
                fmt = QtGui.QImage.Format.Format_RGBX8888
            else:
                fmt = QtGui.QImage.Format.Format_RGBA8888
    elif is_indexed8:
        fmt = QtGui.QImage.Format.Format_Indexed8
        if lut.ndim == 1 or lut.shape[1] == 1:
            ctbl = [QtGui.qRgb(x, x, x) for x in lut.ravel().tolist()]
        elif lut.shape[1] == 3:
            ctbl = [QtGui.qRgb(*rgb) for rgb in lut.tolist()]
        elif lut.shape[1] == 4:
            ctbl = [QtGui.qRgba(*rgba) for rgba in lut.tolist()]
    elif can_grayscale16:
        fmt = QtGui.QImage.Format.Format_Grayscale16
    elif is_rgba64:
        fmt = QtGui.QImage.Format.Format_RGBA64
    if fmt is None:
        raise ValueError('unsupported image type')
    qimage = functions.ndarray_to_qimage(image, fmt)
    if ctbl is not None:
        qimage.setColorTable(ctbl)
    return qimage