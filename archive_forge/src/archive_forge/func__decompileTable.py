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
def _decompileTable(self, tag):
    """Fetch table data, decompile it, and store it inside self.ttFont."""
    tag = Tag(tag)
    if tag not in self.tables:
        raise TTLibError('missing required table: %s' % tag)
    if self.ttFont.isLoaded(tag):
        return
    data = self.tables[tag].data
    if tag == 'loca':
        tableClass = WOFF2LocaTable
    elif tag == 'glyf':
        tableClass = WOFF2GlyfTable
    elif tag == 'hmtx':
        tableClass = WOFF2HmtxTable
    else:
        tableClass = getTableClass(tag)
    table = tableClass(tag)
    self.ttFont.tables[tag] = table
    table.decompile(data, self.ttFont)