import math
import warnings
from ctypes import c_void_p, c_int32, byref, c_byte
from pyglet.font import base
import pyglet.image
from pyglet.libs.darwin import cocoapy, kCTFontURLAttribute, CGFloat
def _lookup_font_with_family_and_traits(self, family, traits):
    if family not in self._loaded_CGFont_table:
        return None
    fonts = self._loaded_CGFont_table[family]
    if not fonts:
        return None
    if traits in fonts:
        return fonts[traits]
    for t, f in fonts.items():
        if traits & t:
            return f
    if 0 in fonts:
        return fonts[0]
    return list(fonts.values())[0]