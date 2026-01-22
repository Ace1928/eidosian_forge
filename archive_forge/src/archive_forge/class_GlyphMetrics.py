import io
import sys
from ctypes import *
import ctypes.util
import struct
from freetype.raw import *
class GlyphMetrics(object):
    """

    A structure used to model the metrics of a single glyph. The values are
    expressed in 26.6 fractional pixel format; if the flag FT_LOAD_NO_SCALE has
    been used while loading the glyph, values are expressed in font units
    instead.

    **Note**

    If not disabled with FT_LOAD_NO_HINTING, the values represent dimensions of
    the hinted glyph (in case hinting is applicable).

    Stroking a glyph with an outside border does not increase ‘horiAdvance’ or
    ‘vertAdvance’; you have to manually adjust these values to account for the
    added width and height.
    """

    def __init__(self, metrics):
        """
        Create a new GlyphMetrics object.

        :param metrics: a FT_Glyph_Metrics
        """
        self._FT_Glyph_Metrics = metrics
    width = property(lambda self: self._FT_Glyph_Metrics.width, doc="The glyph's width.")
    height = property(lambda self: self._FT_Glyph_Metrics.height, doc="The glyph's height.")
    horiBearingX = property(lambda self: self._FT_Glyph_Metrics.horiBearingX, doc='Left side bearing for horizontal layout.')
    horiBearingY = property(lambda self: self._FT_Glyph_Metrics.horiBearingY, doc='Top side bearing for horizontal layout.')
    horiAdvance = property(lambda self: self._FT_Glyph_Metrics.horiAdvance, doc='Advance width for horizontal layout.')
    vertBearingX = property(lambda self: self._FT_Glyph_Metrics.vertBearingX, doc='Left side bearing for vertical layout.')
    vertBearingY = property(lambda self: self._FT_Glyph_Metrics.vertBearingY, doc='Top side bearing for vertical layout. Larger positive values\n                mean further below the vertical glyph origin.')
    vertAdvance = property(lambda self: self._FT_Glyph_Metrics.vertAdvance, doc='Advance height for vertical layout. Positive values mean the\n                glyph has a positive advance downward.')