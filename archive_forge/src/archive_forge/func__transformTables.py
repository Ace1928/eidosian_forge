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
def _transformTables(self):
    """Return transformed font data."""
    transformedTables = self.flavorData.transformedTables
    for tag, entry in self.tables.items():
        data = None
        if tag in transformedTables:
            data = self.transformTable(tag)
            if data is not None:
                entry.transformed = True
        if data is None:
            if tag == 'glyf':
                transformedTables.discard('loca')
            data = entry.data
            entry.transformed = False
        entry.offset = self.nextTableOffset
        entry.saveData(self.transformBuffer, data)
        self.nextTableOffset += entry.length
    self.writeMasterChecksum()
    fontData = self.transformBuffer.getvalue()
    return fontData