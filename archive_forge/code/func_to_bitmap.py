import io
import sys
from ctypes import *
import ctypes.util
import struct
from freetype.raw import *
def to_bitmap(self, mode, origin, destroy=False):
    """
        Convert a given glyph object to a bitmap glyph object.

        :param mode: An enumeration that describes how the data is rendered.

        :param origin: A pointer to a vector used to translate the glyph image
                       before rendering. Can be 0 (if no translation). The origin is
                       expressed in 26.6 pixels.

                       We also detect a plain vector and make a pointer out of it,
                       if that's the case.

        :param destroy: A boolean that indicates that the original glyph image
                        should be destroyed by this function. It is never destroyed
                        in case of error.

        **Note**:

          This function does nothing if the glyph format isn't scalable.

          The glyph image is translated with the 'origin' vector before
          rendering.

          The first parameter is a pointer to an FT_Glyph handle, that will be
          replaced by this function (with newly allocated data). Typically, you
          would use (omitting error handling):
        """
    if type(origin) == FT_Vector:
        error = FT_Glyph_To_Bitmap(byref(self._FT_Glyph), mode, byref(origin), destroy)
    else:
        error = FT_Glyph_To_Bitmap(byref(self._FT_Glyph), mode, origin, destroy)
    if error:
        raise FT_Exception(error)
    return BitmapGlyph(self._FT_Glyph)