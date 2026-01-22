import io
import sys
from ctypes import *
import ctypes.util
import struct
from freetype.raw import *
class Charmap(object):
    """
    FT_Charmap wrapper.

    A handle to a given character map. A charmap is used to translate character
    codes in a given encoding into glyph indexes for its parent's face. Some
    font formats may provide several charmaps per font.

    Each face object owns zero or more charmaps, but only one of them can be
    'active' and used by FT_Get_Char_Index or FT_Load_Char.

    The list of available charmaps in a face is available through the
    'face.num_charmaps' and 'face.charmaps' fields of FT_FaceRec.

    The currently active charmap is available as 'face.charmap'. You should
    call FT_Set_Charmap to change it.

    **Note**:

      When a new face is created (either through FT_New_Face or FT_Open_Face),
      the library looks for a Unicode charmap within the list and automatically
      activates it.

    **See also**:

      See FT_CharMapRec for the publicly accessible fields of a given character
      map.
    """

    def __init__(self, charmap):
        """
        Create a new Charmap object.

        Parameters:
        -----------
        charmap : a FT_Charmap
        """
        self._FT_Charmap = charmap
    encoding = property(lambda self: self._FT_Charmap.contents.encoding, doc='An FT_Encoding tag identifying the charmap. Use this with\n                  FT_Select_Charmap.')
    platform_id = property(lambda self: self._FT_Charmap.contents.platform_id, doc='An ID number describing the platform for the following\n                     encoding ID. This comes directly from the TrueType\n                     specification and should be emulated for other\n                     formats.')
    encoding_id = property(lambda self: self._FT_Charmap.contents.encoding_id, doc='A platform specific encoding number. This also comes from\n                     the TrueType specification and should be emulated\n                     similarly.')

    def _get_encoding_name(self):
        encoding = self.encoding
        for key, value in FT_ENCODINGS.items():
            if encoding == value:
                return key
        return 'Unknown encoding'
    encoding_name = property(_get_encoding_name, doc='A platform specific encoding name. This also comes from\n                     the TrueType specification and should be emulated\n                     similarly.')

    def _get_index(self):
        return FT_Get_Charmap_Index(self._FT_Charmap)
    index = property(_get_index, doc="The index into the array of character maps within the face to\n               which 'charmap' belongs. If an error occurs, -1 is returned.")

    def _get_cmap_language_id(self):
        return FT_Get_CMap_Language_ID(self._FT_Charmap)
    cmap_language_id = property(_get_cmap_language_id, doc="The language ID of 'charmap'. If 'charmap' doesn't\n                          belong to a TrueType/sfnt face, just return 0 as the\n                          default value.")

    def _get_cmap_format(self):
        return FT_Get_CMap_Format(self._FT_Charmap)
    cmap_format = property(_get_cmap_format, doc="The format of 'charmap'. If 'charmap' doesn't belong to a\n                     TrueType/sfnt face, return -1.")