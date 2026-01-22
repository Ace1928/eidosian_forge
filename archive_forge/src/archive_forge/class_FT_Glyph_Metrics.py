from ctypes import *
from .base import FontException
import pyglet.lib
class FT_Glyph_Metrics(Structure):
    _fields_ = [('width', FT_Pos), ('height', FT_Pos), ('horiBearingX', FT_Pos), ('horiBearingY', FT_Pos), ('horiAdvance', FT_Pos), ('vertBearingX', FT_Pos), ('vertBearingY', FT_Pos), ('vertAdvance', FT_Pos)]

    def dump(self):
        for name, type in self._fields_:
            print('FT_Glyph_Metrics', name, repr(getattr(self, name)))