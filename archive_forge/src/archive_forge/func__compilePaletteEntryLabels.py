from fontTools.misc.textTools import bytesjoin, safeEval
from . import DefaultTable
import array
from collections import namedtuple
import struct
import sys
def _compilePaletteEntryLabels(self):
    if self.version == 0 or all((l == self.NO_NAME_ID for l in self.paletteEntryLabels)):
        return b''
    assert len(self.paletteEntryLabels) == self.numPaletteEntries
    result = bytesjoin([struct.pack('>H', label) for label in self.paletteEntryLabels])
    assert len(result) == 2 * self.numPaletteEntries
    return result