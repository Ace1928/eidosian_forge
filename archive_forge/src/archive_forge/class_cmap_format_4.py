from fontTools.misc.textTools import bytesjoin, safeEval, readHex
from fontTools.misc.encodingTools import getEncoding
from fontTools.ttLib import getSearchRange
from fontTools.unicode import Unicode
from . import DefaultTable
import sys
import struct
import array
import logging
class cmap_format_4(CmapSubtable):

    def decompile(self, data, ttFont):
        if data is not None and ttFont is not None:
            self.decompileHeader(data, ttFont)
        else:
            assert data is None and ttFont is None, 'Need both data and ttFont arguments'
        data = self.data
        segCountX2, searchRange, entrySelector, rangeShift = struct.unpack('>4H', data[:8])
        data = data[8:]
        segCount = segCountX2 // 2
        allCodes = array.array('H')
        allCodes.frombytes(data)
        self.data = data = None
        if sys.byteorder != 'big':
            allCodes.byteswap()
        endCode = allCodes[:segCount]
        allCodes = allCodes[segCount + 1:]
        startCode = allCodes[:segCount]
        allCodes = allCodes[segCount:]
        idDelta = allCodes[:segCount]
        allCodes = allCodes[segCount:]
        idRangeOffset = allCodes[:segCount]
        glyphIndexArray = allCodes[segCount:]
        lenGIArray = len(glyphIndexArray)
        charCodes = []
        gids = []
        for i in range(len(startCode) - 1):
            start = startCode[i]
            delta = idDelta[i]
            rangeOffset = idRangeOffset[i]
            partial = rangeOffset // 2 - start + i - len(idRangeOffset)
            rangeCharCodes = list(range(startCode[i], endCode[i] + 1))
            charCodes.extend(rangeCharCodes)
            if rangeOffset == 0:
                gids.extend([charCode + delta & 65535 for charCode in rangeCharCodes])
            else:
                for charCode in rangeCharCodes:
                    index = charCode + partial
                    assert index < lenGIArray, 'In format 4 cmap, range (%d), the calculated index (%d) into the glyph index array is not less than the length of the array (%d) !' % (i, index, lenGIArray)
                    if glyphIndexArray[index] != 0:
                        glyphID = glyphIndexArray[index] + delta
                    else:
                        glyphID = 0
                    gids.append(glyphID & 65535)
        self.cmap = _make_map(self.ttFont, charCodes, gids)

    def compile(self, ttFont):
        if self.data:
            return struct.pack('>HHH', self.format, self.length, self.language) + self.data
        charCodes = list(self.cmap.keys())
        if not charCodes:
            startCode = [65535]
            endCode = [65535]
        else:
            charCodes.sort()
            names = [self.cmap[code] for code in charCodes]
            nameMap = ttFont.getReverseGlyphMap()
            try:
                gids = [nameMap[name] for name in names]
            except KeyError:
                nameMap = ttFont.getReverseGlyphMap(rebuild=True)
                try:
                    gids = [nameMap[name] for name in names]
                except KeyError:
                    gids = []
                    for name in names:
                        try:
                            gid = nameMap[name]
                        except KeyError:
                            try:
                                if name[:3] == 'gid':
                                    gid = int(name[3:])
                                else:
                                    gid = ttFont.getGlyphID(name)
                            except:
                                raise KeyError(name)
                        gids.append(gid)
            cmap = {}
            for code, gid in zip(charCodes, gids):
                cmap[code] = gid
            lastCode = charCodes[0]
            endCode = []
            startCode = [lastCode]
            for charCode in charCodes[1:]:
                if charCode == lastCode + 1:
                    lastCode = charCode
                    continue
                start, end = splitRange(startCode[-1], lastCode, cmap)
                startCode.extend(start)
                endCode.extend(end)
                startCode.append(charCode)
                lastCode = charCode
            start, end = splitRange(startCode[-1], lastCode, cmap)
            startCode.extend(start)
            endCode.extend(end)
            startCode.append(65535)
            endCode.append(65535)
        idDelta = []
        idRangeOffset = []
        glyphIndexArray = []
        for i in range(len(endCode) - 1):
            indices = []
            for charCode in range(startCode[i], endCode[i] + 1):
                indices.append(cmap[charCode])
            if indices == list(range(indices[0], indices[0] + len(indices))):
                idDelta.append((indices[0] - startCode[i]) % 65536)
                idRangeOffset.append(0)
            else:
                idDelta.append(0)
                idRangeOffset.append(2 * (len(endCode) + len(glyphIndexArray) - i))
                glyphIndexArray.extend(indices)
        idDelta.append(1)
        idRangeOffset.append(0)
        segCount = len(endCode)
        segCountX2 = segCount * 2
        searchRange, entrySelector, rangeShift = getSearchRange(segCount, 2)
        charCodeArray = array.array('H', endCode + [0] + startCode)
        idDeltaArray = array.array('H', idDelta)
        restArray = array.array('H', idRangeOffset + glyphIndexArray)
        if sys.byteorder != 'big':
            charCodeArray.byteswap()
        if sys.byteorder != 'big':
            idDeltaArray.byteswap()
        if sys.byteorder != 'big':
            restArray.byteswap()
        data = charCodeArray.tobytes() + idDeltaArray.tobytes() + restArray.tobytes()
        length = struct.calcsize(cmap_format_4_format) + len(data)
        header = struct.pack(cmap_format_4_format, self.format, length, self.language, segCountX2, searchRange, entrySelector, rangeShift)
        return header + data

    def fromXML(self, name, attrs, content, ttFont):
        self.language = safeEval(attrs['language'])
        if not hasattr(self, 'cmap'):
            self.cmap = {}
        cmap = self.cmap
        for element in content:
            if not isinstance(element, tuple):
                continue
            nameMap, attrsMap, dummyContent = element
            if nameMap != 'map':
                assert 0, 'Unrecognized keyword in cmap subtable'
            cmap[safeEval(attrsMap['code'])] = attrsMap['name']