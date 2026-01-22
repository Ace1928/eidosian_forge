from . import _check_status, _keepref, cairo, constants, ffi
from .matrix import Matrix
from .surfaces import Surface
class SolidPattern(Pattern):
    """Creates a new pattern corresponding to a solid color.
    The color and alpha components are in the range 0 to 1.
    If the values passed in are outside that range, they will be clamped.

    :param red: Red component of the color.
    :param green: Green component of the color.
    :param blue: Blue component of the color.
    :param alpha:
        Alpha component of the color.
        1 (the default) is opaque, 0 fully transparent.
    :type red: float
    :type green: float
    :type blue: float
    :type alpha: float

    """

    def __init__(self, red, green, blue, alpha=1):
        Pattern.__init__(self, cairo.cairo_pattern_create_rgba(red, green, blue, alpha))

    def get_rgba(self):
        """Returns the solid pattern’s color.

        :returns: a ``(red, green, blue, alpha)`` tuple of floats.

        """
        rgba = ffi.new('double[4]')
        _check_status(cairo.cairo_pattern_get_rgba(self._pointer, rgba + 0, rgba + 1, rgba + 2, rgba + 3))
        return tuple(rgba)