from fontTools.misc import sstruct
from fontTools.misc import psCharStrings
from fontTools.misc.arrayTools import unionRect, intRect
from fontTools.misc.textTools import (
from fontTools.ttLib import TTFont
from fontTools.ttLib.tables.otBase import OTTableWriter
from fontTools.ttLib.tables.otBase import OTTableReader
from fontTools.ttLib.tables import otTables as ot
from io import BytesIO
import struct
import logging
import re
class PrivateDict(BaseDict):
    defaults = buildDefaults(privateDictOperators)
    converters = buildConverters(privateDictOperators)
    order = buildOrder(privateDictOperators)
    decompilerClass = PrivateDictDecompiler
    compilerClass = PrivateDictCompiler

    def __init__(self, strings=None, file=None, offset=None, isCFF2=None, vstore=None):
        super(PrivateDict, self).__init__(strings, file, offset, isCFF2=isCFF2)
        self.vstore = vstore
        if isCFF2:
            self.defaults = buildDefaults(privateDictOperators2)
            self.order = buildOrder(privateDictOperators2)
            self.nominalWidthX = self.defaultWidthX = None
        else:
            self.defaults = buildDefaults(privateDictOperators)
            self.order = buildOrder(privateDictOperators)

    @property
    def in_cff2(self):
        return self._isCFF2

    def getNumRegions(self, vi=None):
        if vi is None:
            if hasattr(self, 'vsindex'):
                vi = self.vsindex
            else:
                vi = 0
        numRegions = self.vstore.getNumRegions(vi)
        return numRegions