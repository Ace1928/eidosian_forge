from . import _check_status, _keepref, cairo, constants, ffi
from .matrix import Matrix
from .surfaces import Surface
class RadialGradient(Gradient):
    """Creates a new radial gradient pattern between the two circles
    defined by (cx0, cy0, radius0) and (cx1, cy1, radius1).
    Before using the gradient pattern, a number of color stops
    should be defined using :meth:`~Gradient.add_color_stop_rgba`.

    Note: The coordinates here are in pattern space.
    For a new pattern, pattern space is identical to user space,
    but the relationship between the spaces can be changed
    with :meth:`~Pattern.set_matrix`.

    :param cx0: X coordinate of the start circle.
    :param cy0: Y coordinate of the start circle.
    :param radius0: Radius of the start circle.
    :param cx1: X coordinate of the end circle.
    :param cy1: Y coordinate of the end circle.
    :param radius1: Y coordinate of the end circle.
    :type cx0: float
    :type cy0: float
    :type radius0: float
    :type cx1: float
    :type cy1: float
    :type radius1: float

    """

    def __init__(self, cx0, cy0, radius0, cx1, cy1, radius1):
        Pattern.__init__(self, cairo.cairo_pattern_create_radial(cx0, cy0, radius0, cx1, cy1, radius1))

    def get_radial_circles(self):
        """Return this radial gradient’s endpoint circles,
        each specified as a center coordinate and a radius.

        :returns: A ``(cx0, cy0, radius0, cx1, cy1, radius1)`` tuple of floats.

        """
        circles = ffi.new('double[6]')
        _check_status(cairo.cairo_pattern_get_radial_circles(self._pointer, circles + 0, circles + 1, circles + 2, circles + 3, circles + 4, circles + 5))
        return tuple(circles)