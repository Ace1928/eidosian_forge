from . import _check_status, _keepref, cairo, constants, ffi
from .fonts import FontFace, FontOptions, ScaledFont, _encode_string
from .matrix import Matrix
from .patterns import Pattern
from .surfaces import Surface
def user_to_device_distance(self, dx, dy):
    """Transform a distance vector from user space to device space.
        This method is similar to :meth:`Context.user_to_device`
        except that the translation components of the CTM
        will be ignored when transforming ``(dx, dy)``.

        :param dx: X component of a distance vector.
        :param dy: Y component of a distance vector.
        :type x: float
        :type y: float
        :returns: A ``(device_dx, device_dy)`` tuple of floats.

        """
    xy = ffi.new('double[2]', [dx, dy])
    cairo.cairo_user_to_device_distance(self._pointer, xy + 0, xy + 1)
    self._check_status()
    return tuple(xy)