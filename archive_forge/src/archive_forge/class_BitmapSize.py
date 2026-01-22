import io
import sys
from ctypes import *
import ctypes.util
import struct
from freetype.raw import *
class BitmapSize(object):
    """
    FT_Bitmap_Size wrapper

    This structure models the metrics of a bitmap strike (i.e., a set of glyphs
    for a given point size and resolution) in a bitmap font. It is used for the
    'available_sizes' field of Face.

    **Note**

    Windows FNT: The nominal size given in a FNT font is not reliable. Thus
    when the driver finds it incorrect, it sets 'size' to some calculated
    values and sets 'x_ppem' and 'y_ppem' to the pixel width and height given
    in the font, respectively.

    TrueType embedded bitmaps: 'size', 'width', and 'height' values are not
    contained in the bitmap strike itself. They are computed from the global
    font parameters.
    """

    def __init__(self, size):
        """
        Create a new SizeMetrics object.

        :param size: a FT_Bitmap_Size
        """
        self._FT_Bitmap_Size = size
    height = property(lambda self: self._FT_Bitmap_Size.height, doc='The vertical distance, in pixels, between two consecutive\n                baselines. It is always positive.')
    width = property(lambda self: self._FT_Bitmap_Size.width, doc='The average width, in pixels, of all glyphs in the strike.')
    size = property(lambda self: self._FT_Bitmap_Size.size, doc='The nominal size of the strike in 26.6 fractional points. This\n              field is not very useful.')
    x_ppem = property(lambda self: self._FT_Bitmap_Size.x_ppem, doc='The horizontal ppem (nominal width) in 26.6 fractional\n                pixels.')
    y_ppem = property(lambda self: self._FT_Bitmap_Size.y_ppem, doc='The vertical ppem (nominal width) in 26.6 fractional\n                pixels.')