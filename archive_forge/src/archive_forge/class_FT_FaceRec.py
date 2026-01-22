from ctypes import *
from .base import FontException
import pyglet.lib
class FT_FaceRec(Structure):
    _fields_ = [('num_faces', FT_Long), ('face_index', FT_Long), ('face_flags', FT_Long), ('style_flags', FT_Long), ('num_glyphs', FT_Long), ('family_name', FT_String_Ptr), ('style_name', FT_String_Ptr), ('num_fixed_sizes', FT_Int), ('available_sizes', POINTER(FT_Bitmap_Size)), ('num_charmaps', FT_Int), ('charmaps', c_void_p), ('generic', FT_Generic), ('bbox', FT_BBox), ('units_per_EM', FT_UShort), ('ascender', FT_Short), ('descender', FT_Short), ('height', FT_Short), ('max_advance_width', FT_Short), ('max_advance_height', FT_Short), ('underline_position', FT_Short), ('underline_thickness', FT_Short), ('glyph', FT_GlyphSlot), ('size', FT_Size), ('charmap', c_void_p), ('driver', c_void_p), ('memory', c_void_p), ('stream', c_void_p), ('sizes_list', c_void_p), ('autohint', FT_Generic), ('extensions', c_void_p), ('internal', c_void_p)]

    def dump(self):
        for name, type in self._fields_:
            print('FT_FaceRec', name, repr(getattr(self, name)))

    def has_kerning(self):
        return self.face_flags & FT_FACE_FLAG_KERNING