from ctypes import *
from .base import FontException
import pyglet.lib
class FT_GlyphSlotRec(Structure):
    _fields_ = [('library', FT_Library), ('face', c_void_p), ('next', c_void_p), ('reserved', FT_UInt), ('generic', FT_Generic), ('metrics', FT_Glyph_Metrics), ('linearHoriAdvance', FT_Fixed), ('linearVertAdvance', FT_Fixed), ('advance', FT_Vector), ('format', FT_Glyph_Format), ('bitmap', FT_Bitmap), ('bitmap_left', FT_Int), ('bitmap_top', FT_Int), ('outline', FT_Outline), ('num_subglyphs', FT_UInt), ('subglyphs', FT_SubGlyph), ('control_data', c_void_p), ('control_len', c_long), ('lsb_delta', FT_Pos), ('rsb_delta', FT_Pos), ('other', c_void_p), ('internal', c_void_p)]