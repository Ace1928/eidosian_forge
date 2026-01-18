from . import _check_status, _keepref, cairo, constants, ffi
from .fonts import FontFace, FontOptions, ScaledFont, _encode_string
from .matrix import Matrix
from .patterns import Pattern
from .surfaces import Surface
def user_to_device(self, x, y):
    """Transform a coordinate from user space to device space
        by multiplying the given point
        by the current transformation matrix (CTM).

        :param x: X position.
        :param y: Y position.
        :type x: float
        :type y: float
        :returns: A ``(device_x, device_y)`` tuple of floats.

        """
    xy = ffi.new('double[2]', [x, y])
    cairo.cairo_user_to_device(self._pointer, xy + 0, xy + 1)
    self._check_status()
    return tuple(xy)