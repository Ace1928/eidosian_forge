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
class FDArrayIndexCompiler(IndexCompiler):
    """Helper class for writing the
    `Font DICT INDEX <https://docs.microsoft.com/en-us/typography/opentype/spec/cff2#10-font-dict-index-font-dicts-and-fdselect>`_
    to binary."""

    def getItems(self, items, strings):
        out = []
        for item in items:
            out.append(item.getCompiler(strings, self))
        return out

    def getChildren(self, strings):
        children = []
        for fontDict in self.items:
            children.extend(fontDict.getChildren(strings))
        return children

    def toFile(self, file):
        offsets = self.getOffsets()
        if self.isCFF2:
            writeCard32(file, len(self.items))
        else:
            writeCard16(file, len(self.items))
        offSize = calcOffSize(offsets[-1])
        writeCard8(file, offSize)
        offSize = -offSize
        pack = struct.pack
        for offset in offsets:
            binOffset = pack('>l', offset)[offSize:]
            assert len(binOffset) == -offSize
            file.write(binOffset)
        for item in self.items:
            if hasattr(item, 'toFile'):
                item.toFile(file)
            else:
                file.write(item)

    def setPos(self, pos, endPos):
        self.parent.rawDict['FDArray'] = pos