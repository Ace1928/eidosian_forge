from fontTools.misc.textTools import bytesjoin
from fontTools.misc import sstruct
from . import E_B_D_T_
from .BitmapGlyphMetrics import (
from .E_B_D_T_ import (
import struct
class ColorBitmapGlyph(BitmapGlyph):
    fileExtension = '.png'
    xmlDataFunctions = _removeUnsupportedForColor(BitmapGlyph.xmlDataFunctions)