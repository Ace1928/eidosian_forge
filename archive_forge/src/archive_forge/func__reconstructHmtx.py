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
def _reconstructHmtx(self, data):
    """Return reconstructed hmtx table data."""
    if 'glyf' in self.flavorData.transformedTables:
        tableDependencies = ('maxp', 'hhea', 'glyf')
    else:
        tableDependencies = ('maxp', 'head', 'hhea', 'loca', 'glyf')
    for tag in tableDependencies:
        self._decompileTable(tag)
    hmtxTable = self.ttFont['hmtx'] = WOFF2HmtxTable()
    hmtxTable.reconstruct(data, self.ttFont)
    data = hmtxTable.compile(self.ttFont)
    return data