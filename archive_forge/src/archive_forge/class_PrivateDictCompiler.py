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
class PrivateDictCompiler(DictCompiler):
    maxBlendStack = maxStackLimit
    opcodes = buildOpcodeDict(privateDictOperators)

    def setPos(self, pos, endPos):
        size = endPos - pos
        self.parent.rawDict['Private'] = (size, pos)
        self.pos = pos

    def getChildren(self, strings):
        children = []
        if hasattr(self.dictObj, 'Subrs'):
            children.append(self.dictObj.Subrs.getCompiler(strings, self))
        return children