from . import _check_status, _keepref, cairo, constants, ffi
from .matrix import Matrix
def text_extents(self, text):
    """Returns the extents for a string of text.

        The extents describe a user-space rectangle
        that encloses the "inked" portion of the text,
        (as it would be drawn by :meth:`Context.show_text`).
        Additionally, the ``x_advance`` and ``y_advance`` values
        indicate the amount by which the current point would be advanced
        by :meth:`Context.show_text`.

        :param text: The text to measure, as an Unicode or UTF-8 string.
        :returns:
            A ``(x_bearing, y_bearing, width, height, x_advance, y_advance)``
            tuple of floats.
            See :meth:`Context.text_extents` for details.

        """
    extents = ffi.new('cairo_text_extents_t *')
    cairo.cairo_scaled_font_text_extents(self._pointer, _encode_string(text), extents)
    self._check_status()
    return (extents.x_bearing, extents.y_bearing, extents.width, extents.height, extents.x_advance, extents.y_advance)