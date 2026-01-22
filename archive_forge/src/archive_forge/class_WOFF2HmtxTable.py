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
class WOFF2HmtxTable(getTableClass('hmtx')):

    def __init__(self, tag=None):
        self.tableTag = Tag(tag or 'hmtx')

    def reconstruct(self, data, ttFont):
        flags, = struct.unpack('>B', data[:1])
        data = data[1:]
        if flags & 252 != 0:
            raise TTLibError("Bits 2-7 of '%s' flags are reserved" % self.tableTag)
        hasLsbArray = flags & 1 == 0
        hasLeftSideBearingArray = flags & 2 == 0
        if hasLsbArray and hasLeftSideBearingArray:
            raise TTLibError("either bits 0 or 1 (or both) must set in transformed '%s' flags" % self.tableTag)
        glyfTable = ttFont['glyf']
        headerTable = ttFont['hhea']
        glyphOrder = glyfTable.glyphOrder
        numGlyphs = len(glyphOrder)
        numberOfHMetrics = min(int(headerTable.numberOfHMetrics), numGlyphs)
        assert len(data) >= 2 * numberOfHMetrics
        advanceWidthArray = array.array('H', data[:2 * numberOfHMetrics])
        if sys.byteorder != 'big':
            advanceWidthArray.byteswap()
        data = data[2 * numberOfHMetrics:]
        if hasLsbArray:
            assert len(data) >= 2 * numberOfHMetrics
            lsbArray = array.array('h', data[:2 * numberOfHMetrics])
            if sys.byteorder != 'big':
                lsbArray.byteswap()
            data = data[2 * numberOfHMetrics:]
        else:
            lsbArray = array.array('h')
            for i, glyphName in enumerate(glyphOrder):
                if i >= numberOfHMetrics:
                    break
                glyph = glyfTable[glyphName]
                xMin = getattr(glyph, 'xMin', 0)
                lsbArray.append(xMin)
        numberOfSideBearings = numGlyphs - numberOfHMetrics
        if hasLeftSideBearingArray:
            assert len(data) >= 2 * numberOfSideBearings
            leftSideBearingArray = array.array('h', data[:2 * numberOfSideBearings])
            if sys.byteorder != 'big':
                leftSideBearingArray.byteswap()
            data = data[2 * numberOfSideBearings:]
        else:
            leftSideBearingArray = array.array('h')
            for i, glyphName in enumerate(glyphOrder):
                if i < numberOfHMetrics:
                    continue
                glyph = glyfTable[glyphName]
                xMin = getattr(glyph, 'xMin', 0)
                leftSideBearingArray.append(xMin)
        if data:
            raise TTLibError("too much '%s' table data" % self.tableTag)
        self.metrics = {}
        for i in range(numberOfHMetrics):
            glyphName = glyphOrder[i]
            advanceWidth, lsb = (advanceWidthArray[i], lsbArray[i])
            self.metrics[glyphName] = (advanceWidth, lsb)
        lastAdvance = advanceWidthArray[-1]
        for i in range(numberOfSideBearings):
            glyphName = glyphOrder[i + numberOfHMetrics]
            self.metrics[glyphName] = (lastAdvance, leftSideBearingArray[i])

    def transform(self, ttFont):
        glyphOrder = ttFont.getGlyphOrder()
        glyf = ttFont['glyf']
        hhea = ttFont['hhea']
        numberOfHMetrics = hhea.numberOfHMetrics
        hasLsbArray = False
        for i in range(numberOfHMetrics):
            glyphName = glyphOrder[i]
            lsb = self.metrics[glyphName][1]
            if lsb != getattr(glyf[glyphName], 'xMin', 0):
                hasLsbArray = True
                break
        hasLeftSideBearingArray = False
        for i in range(numberOfHMetrics, len(glyphOrder)):
            glyphName = glyphOrder[i]
            lsb = self.metrics[glyphName][1]
            if lsb != getattr(glyf[glyphName], 'xMin', 0):
                hasLeftSideBearingArray = True
                break
        if hasLsbArray and hasLeftSideBearingArray:
            return
        flags = 0
        if not hasLsbArray:
            flags |= 1 << 0
        if not hasLeftSideBearingArray:
            flags |= 1 << 1
        data = struct.pack('>B', flags)
        advanceWidthArray = array.array('H', [self.metrics[glyphName][0] for i, glyphName in enumerate(glyphOrder) if i < numberOfHMetrics])
        if sys.byteorder != 'big':
            advanceWidthArray.byteswap()
        data += advanceWidthArray.tobytes()
        if hasLsbArray:
            lsbArray = array.array('h', [self.metrics[glyphName][1] for i, glyphName in enumerate(glyphOrder) if i < numberOfHMetrics])
            if sys.byteorder != 'big':
                lsbArray.byteswap()
            data += lsbArray.tobytes()
        if hasLeftSideBearingArray:
            leftSideBearingArray = array.array('h', [self.metrics[glyphOrder[i]][1] for i in range(numberOfHMetrics, len(glyphOrder))])
            if sys.byteorder != 'big':
                leftSideBearingArray.byteswap()
            data += leftSideBearingArray.tobytes()
        return data