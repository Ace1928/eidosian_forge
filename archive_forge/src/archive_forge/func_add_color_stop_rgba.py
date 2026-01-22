from . import _check_status, _keepref, cairo, constants, ffi
from .matrix import Matrix
from .surfaces import Surface
def add_color_stop_rgba(self, offset, red, green, blue, alpha=1):
    """Adds a translucent color stop to a gradient pattern.

        The offset specifies the location along the gradient's control vector.
        For example,
        a linear gradient's control vector is from (x0,y0) to (x1,y1)
        while a radial gradient's control vector is
        from any point on the start circle
        to the corresponding point on the end circle.

        If two (or more) stops are specified with identical offset values,
        they will be sorted
        according to the order in which the stops are added
        (stops added earlier before stops added later).
        This can be useful for reliably making sharp color transitions
        instead of the typical blend.

        The color components and offset are in the range 0 to 1.
        If the values passed in are outside that range, they will be clamped.

        :param offset: Location along the gradient's control vector
        :param red: Red component of the color.
        :param green: Green component of the color.
        :param blue: Blue component of the color.
        :param alpha:
            Alpha component of the color.
            1 (the default) is opaque, 0 fully transparent.
        :type offset: float
        :type red: float
        :type green: float
        :type blue: float
        :type alpha: float

        """
    cairo.cairo_pattern_add_color_stop_rgba(self._pointer, offset, red, green, blue, alpha)
    self._check_status()