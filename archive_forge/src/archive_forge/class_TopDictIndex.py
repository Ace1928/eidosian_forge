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
class TopDictIndex(Index):
    """This index represents the array of ``TopDict`` structures in the font
    (again, usually only one entry is present). Hence the following calls are
    equivalent:

    .. code:: python

            tt["CFF "].cff[0]
            # <fontTools.cffLib.TopDict object at 0x102ed6e50>
            tt["CFF "].cff.topDictIndex[0]
            # <fontTools.cffLib.TopDict object at 0x102ed6e50>

    """
    compilerClass = TopDictIndexCompiler

    def __init__(self, file=None, cff2GetGlyphOrder=None, topSize=0, isCFF2=None):
        assert (isCFF2 is None) == (file is None)
        self.cff2GetGlyphOrder = cff2GetGlyphOrder
        if file is not None and isCFF2:
            self._isCFF2 = isCFF2
            self.items = []
            name = self.__class__.__name__
            log.log(DEBUG, 'loading %s at %s', name, file.tell())
            self.file = file
            count = 1
            self.items = [None] * count
            self.offsets = [0, topSize]
            self.offsetBase = file.tell()
            file.seek(self.offsetBase + topSize)
            log.log(DEBUG, '    end of %s at %s', name, file.tell())
        else:
            super(TopDictIndex, self).__init__(file, isCFF2=isCFF2)

    def produceItem(self, index, data, file, offset):
        top = TopDict(self.strings, file, offset, self.GlobalSubrs, self.cff2GetGlyphOrder, isCFF2=self._isCFF2)
        top.decompile(data)
        return top

    def toXML(self, xmlWriter):
        for i in range(len(self)):
            xmlWriter.begintag('FontDict', index=i)
            xmlWriter.newline()
            self[i].toXML(xmlWriter)
            xmlWriter.endtag('FontDict')
            xmlWriter.newline()