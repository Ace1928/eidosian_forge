from . import _check_status, _keepref, cairo, constants, ffi
from .fonts import FontFace, FontOptions, ScaledFont, _encode_string
from .matrix import Matrix
from .patterns import Pattern
from .surfaces import Surface
def text_path(self, text):
    """Adds closed paths for text to the current path.
        The generated path if filled,
        achieves an effect similar to that of :meth:`show_text`.

        Text conversion and positioning is done similar to :meth:`show_text`.

        Like :meth:`show_text`,
        after this call the current point is moved to the origin of where
        the next glyph would be placed in this same progression.
        That is, the current point will be at the origin of the final glyph
        offset by its advance values.
        This allows for chaining multiple calls to to :meth:`text_path`
        without having to set current point in between.

        :param text: The text to show, as an Unicode or UTF-8 string.

        .. note::
            The :meth:`text_path` method is part of
            what the cairo designers call the "toy" text API.
            It is convenient for short demos and simple programs,
            but it is not expected to be adequate
            for serious text-using applications.
            See :ref:`fonts` for details,
            and :meth:`glyph_path` for the "real" text path API in cairo.

        """
    cairo.cairo_text_path(self._pointer, _encode_string(text))
    self._check_status()