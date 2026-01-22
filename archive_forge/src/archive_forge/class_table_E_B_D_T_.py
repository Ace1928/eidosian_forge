from fontTools.misc import sstruct
from fontTools.misc.textTools import (
from .BitmapGlyphMetrics import (
from . import DefaultTable
import itertools
import os
import struct
import logging
class table_E_B_D_T_(DefaultTable.DefaultTable):
    locatorName = 'EBLC'

    def getImageFormatClass(self, imageFormat):
        return ebdt_bitmap_classes[imageFormat]

    def decompile(self, data, ttFont):
        sstruct.unpack2(ebdtTableVersionFormat, data, self)
        glyphDict = {}
        locator = ttFont[self.__class__.locatorName]
        self.strikeData = []
        for curStrike in locator.strikes:
            bitmapGlyphDict = {}
            self.strikeData.append(bitmapGlyphDict)
            for indexSubTable in curStrike.indexSubTables:
                dataIter = zip(indexSubTable.names, indexSubTable.locations)
                for curName, curLoc in dataIter:
                    if curLoc in glyphDict:
                        curGlyph = glyphDict[curLoc]
                    else:
                        curGlyphData = data[slice(*curLoc)]
                        imageFormatClass = self.getImageFormatClass(indexSubTable.imageFormat)
                        curGlyph = imageFormatClass(curGlyphData, ttFont)
                        glyphDict[curLoc] = curGlyph
                    bitmapGlyphDict[curName] = curGlyph

    def compile(self, ttFont):
        dataList = []
        dataList.append(sstruct.pack(ebdtTableVersionFormat, self))
        dataSize = len(dataList[0])
        glyphDict = {}
        locator = ttFont[self.__class__.locatorName]
        for curStrike, curGlyphDict in zip(locator.strikes, self.strikeData):
            for curIndexSubTable in curStrike.indexSubTables:
                dataLocations = []
                for curName in curIndexSubTable.names:
                    glyph = curGlyphDict[curName]
                    objectId = id(glyph)
                    if objectId not in glyphDict:
                        data = glyph.compile(ttFont)
                        data = curIndexSubTable.padBitmapData(data)
                        startByte = dataSize
                        dataSize += len(data)
                        endByte = dataSize
                        dataList.append(data)
                        dataLoc = (startByte, endByte)
                        glyphDict[objectId] = dataLoc
                    else:
                        dataLoc = glyphDict[objectId]
                    dataLocations.append(dataLoc)
                curIndexSubTable.locations = dataLocations
        return bytesjoin(dataList)

    def toXML(self, writer, ttFont):
        if ttFont.bitmapGlyphDataFormat in ('row', 'bitwise'):
            locator = ttFont[self.__class__.locatorName]
            for curStrike, curGlyphDict in zip(locator.strikes, self.strikeData):
                for curIndexSubTable in curStrike.indexSubTables:
                    for curName in curIndexSubTable.names:
                        glyph = curGlyphDict[curName]
                        if hasattr(glyph, 'metrics'):
                            glyph.exportMetrics = glyph.metrics
                        else:
                            glyph.exportMetrics = curIndexSubTable.metrics
                        glyph.exportBitDepth = curStrike.bitmapSizeTable.bitDepth
        writer.simpletag('header', [('version', self.version)])
        writer.newline()
        locator = ttFont[self.__class__.locatorName]
        for strikeIndex, bitmapGlyphDict in enumerate(self.strikeData):
            writer.begintag('strikedata', [('index', strikeIndex)])
            writer.newline()
            for curName, curBitmap in bitmapGlyphDict.items():
                curBitmap.toXML(strikeIndex, curName, writer, ttFont)
            writer.endtag('strikedata')
            writer.newline()

    def fromXML(self, name, attrs, content, ttFont):
        if name == 'header':
            self.version = safeEval(attrs['version'])
        elif name == 'strikedata':
            if not hasattr(self, 'strikeData'):
                self.strikeData = []
            strikeIndex = safeEval(attrs['index'])
            bitmapGlyphDict = {}
            for element in content:
                if not isinstance(element, tuple):
                    continue
                name, attrs, content = element
                if name[4:].startswith(_bitmapGlyphSubclassPrefix[4:]):
                    imageFormat = safeEval(name[len(_bitmapGlyphSubclassPrefix):])
                    glyphName = attrs['name']
                    imageFormatClass = self.getImageFormatClass(imageFormat)
                    curGlyph = imageFormatClass(None, None)
                    curGlyph.fromXML(name, attrs, content, ttFont)
                    assert glyphName not in bitmapGlyphDict, "Duplicate glyphs with the same name '%s' in the same strike." % glyphName
                    bitmapGlyphDict[glyphName] = curGlyph
                else:
                    log.warning('%s being ignored by %s', name, self.__class__.__name__)
            if strikeIndex >= len(self.strikeData):
                self.strikeData += [None] * (strikeIndex + 1 - len(self.strikeData))
            assert self.strikeData[strikeIndex] is None, 'Duplicate strike EBDT indices.'
            self.strikeData[strikeIndex] = bitmapGlyphDict