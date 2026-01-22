from io import BytesIO
import sys
import array
import struct
from collections import OrderedDict
from fontTools.misc import sstruct
from fontTools.misc.arrayTools import calcIntBounds
from fontTools.misc.textTools import Tag, bytechr, byteord, bytesjoin, pad
from fontTools.ttLib import (
from fontTools.ttLib.sfnt import (
from fontTools.ttLib.tables import ttProgram, _g_l_y_f
import logging
def _reconstructGlyf(self, data, padding=None):
    """Return recostructed glyf table data, and set the corresponding loca's
        locations. Optionally pad glyph offsets to the specified number of bytes.
        """
    self.ttFont['loca'] = WOFF2LocaTable()
    glyfTable = self.ttFont['glyf'] = WOFF2GlyfTable()
    glyfTable.reconstruct(data, self.ttFont)
    if padding:
        glyfTable.padding = padding
    data = glyfTable.compile(self.ttFont)
    return data