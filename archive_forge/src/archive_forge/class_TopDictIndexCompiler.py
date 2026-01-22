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
class TopDictIndexCompiler(IndexCompiler):
    """Helper class for writing the TopDict to binary."""

    def getItems(self, items, strings):
        out = []
        for item in items:
            out.append(item.getCompiler(strings, self))
        return out

    def getChildren(self, strings):
        children = []
        for topDict in self.items:
            children.extend(topDict.getChildren(strings))
        return children

    def getOffsets(self):
        if self.isCFF2:
            offsets = [0, self.items[0].getDataLength()]
            return offsets
        else:
            return super(TopDictIndexCompiler, self).getOffsets()

    def getDataLength(self):
        if self.isCFF2:
            dataLength = self.items[0].getDataLength()
            return dataLength
        else:
            return super(TopDictIndexCompiler, self).getDataLength()

    def toFile(self, file):
        if self.isCFF2:
            self.items[0].toFile(file)
        else:
            super(TopDictIndexCompiler, self).toFile(file)