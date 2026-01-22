from fontTools.misc.fixedTools import (
from fontTools.misc.roundTools import nearestMultipleShortestRepr, otRound
from fontTools.misc.textTools import bytesjoin, tobytes, tostr, pad, safeEval
from fontTools.ttLib import getSearchRange
from .otBase import (
from .otTables import (
from itertools import zip_longest
from functools import partial
import re
import struct
from typing import Optional
import logging
def _compilePerGlyphLookups(self, table, font):
    if self.perGlyphLookup is None:
        return b''
    numLookups = self._countPerGlyphLookups(table)
    assert len(table.PerGlyphLookups) == numLookups, 'len(AATStateTable.PerGlyphLookups) is %d, but the actions inside the table refer to %d' % (len(table.PerGlyphLookups), numLookups)
    writer = OTTableWriter()
    for lookup in table.PerGlyphLookups:
        lookupWriter = writer.getSubWriter()
        self.perGlyphLookup.write(lookupWriter, font, {}, lookup, None)
        writer.writeSubTable(lookupWriter, offsetSize=4)
    return writer.getAllData()