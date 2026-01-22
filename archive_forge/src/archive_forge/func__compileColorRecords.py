from fontTools.misc.textTools import bytesjoin, safeEval
from . import DefaultTable
import array
from collections import namedtuple
import struct
import sys
def _compileColorRecords(self):
    colorRecords, colorRecordIndices, pool = ([], [], {})
    for palette in self.palettes:
        packedPalette = self._compilePalette(palette)
        if packedPalette in pool:
            index = pool[packedPalette]
        else:
            index = len(colorRecords)
            colorRecords.append(packedPalette)
            pool[packedPalette] = index
        colorRecordIndices.append(struct.pack('>H', index * self.numPaletteEntries))
    return (bytesjoin(colorRecordIndices), bytesjoin(colorRecords))