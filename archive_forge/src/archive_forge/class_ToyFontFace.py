from . import _check_status, _keepref, cairo, constants, ffi
from .matrix import Matrix
class ToyFontFace(FontFace):
    """Creates a font face from a triplet of family, slant, and weight.
    These font faces are used in implementation of cairo’s "toy" font API.

    If family is the zero-length string ``""``,
    the platform-specific default family is assumed.
    The default family then can be queried using :meth:`get_family`.

    The :meth:`Context.select_font_face` method uses this to create font faces.
    See that method for limitations and other details of toy font faces.

    :param family: a font family name, as an Unicode or UTF-8 string.
    :param slant: The :ref:`FONT_SLANT` string for the font face.
    :param weight: The :ref:`FONT_WEIGHT` string for the font face.

    """

    def __init__(self, family='', slant=constants.FONT_SLANT_NORMAL, weight=constants.FONT_WEIGHT_NORMAL):
        FontFace.__init__(self, cairo.cairo_toy_font_face_create(_encode_string(family), slant, weight))

    def get_family(self):
        """Return this font face’s family name."""
        return ffi.string(cairo.cairo_toy_font_face_get_family(self._pointer)).decode('utf8', 'replace')

    def get_slant(self):
        """Return this font face’s :ref:`FONT_SLANT` string."""
        return cairo.cairo_toy_font_face_get_slant(self._pointer)

    def get_weight(self):
        """Return this font face’s :ref:`FONT_WEIGHT` string."""
        return cairo.cairo_toy_font_face_get_weight(self._pointer)