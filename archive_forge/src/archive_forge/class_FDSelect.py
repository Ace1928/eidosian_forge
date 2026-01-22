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
class FDSelect(object):

    def __init__(self, file=None, numGlyphs=None, format=None):
        if file:
            self.format = readCard8(file)
            if self.format == 0:
                from array import array
                self.gidArray = array('B', file.read(numGlyphs)).tolist()
            elif self.format == 3:
                gidArray = [None] * numGlyphs
                nRanges = readCard16(file)
                fd = None
                prev = None
                for i in range(nRanges):
                    first = readCard16(file)
                    if prev is not None:
                        for glyphID in range(prev, first):
                            gidArray[glyphID] = fd
                    prev = first
                    fd = readCard8(file)
                if prev is not None:
                    first = readCard16(file)
                    for glyphID in range(prev, first):
                        gidArray[glyphID] = fd
                self.gidArray = gidArray
            elif self.format == 4:
                gidArray = [None] * numGlyphs
                nRanges = readCard32(file)
                fd = None
                prev = None
                for i in range(nRanges):
                    first = readCard32(file)
                    if prev is not None:
                        for glyphID in range(prev, first):
                            gidArray[glyphID] = fd
                    prev = first
                    fd = readCard16(file)
                if prev is not None:
                    first = readCard32(file)
                    for glyphID in range(prev, first):
                        gidArray[glyphID] = fd
                self.gidArray = gidArray
            else:
                assert False, 'unsupported FDSelect format: %s' % format
        else:
            self.format = format
            self.gidArray = []

    def __len__(self):
        return len(self.gidArray)

    def __getitem__(self, index):
        return self.gidArray[index]

    def __setitem__(self, index, fdSelectValue):
        self.gidArray[index] = fdSelectValue

    def append(self, fdSelectValue):
        self.gidArray.append(fdSelectValue)