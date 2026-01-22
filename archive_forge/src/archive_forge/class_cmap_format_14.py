from fontTools.misc.textTools import bytesjoin, safeEval, readHex
from fontTools.misc.encodingTools import getEncoding
from fontTools.ttLib import getSearchRange
from fontTools.unicode import Unicode
from . import DefaultTable
import sys
import struct
import array
import logging
class cmap_format_14(CmapSubtable):

    def decompileHeader(self, data, ttFont):
        format, length, numVarSelectorRecords = struct.unpack('>HLL', data[:10])
        self.data = data[10:]
        self.length = length
        self.numVarSelectorRecords = numVarSelectorRecords
        self.ttFont = ttFont
        self.language = 255

    def decompile(self, data, ttFont):
        if data is not None and ttFont is not None:
            self.decompileHeader(data, ttFont)
        else:
            assert data is None and ttFont is None, 'Need both data and ttFont arguments'
        data = self.data
        self.cmap = {}
        uvsDict = {}
        recOffset = 0
        for n in range(self.numVarSelectorRecords):
            uvs, defOVSOffset, nonDefUVSOffset = struct.unpack('>3sLL', data[recOffset:recOffset + 11])
            recOffset += 11
            varUVS = cvtToUVS(uvs)
            if defOVSOffset:
                startOffset = defOVSOffset - 10
                numValues, = struct.unpack('>L', data[startOffset:startOffset + 4])
                startOffset += 4
                for r in range(numValues):
                    uv, addtlCnt = struct.unpack('>3sB', data[startOffset:startOffset + 4])
                    startOffset += 4
                    firstBaseUV = cvtToUVS(uv)
                    cnt = addtlCnt + 1
                    baseUVList = list(range(firstBaseUV, firstBaseUV + cnt))
                    glyphList = [None] * cnt
                    localUVList = zip(baseUVList, glyphList)
                    try:
                        uvsDict[varUVS].extend(localUVList)
                    except KeyError:
                        uvsDict[varUVS] = list(localUVList)
            if nonDefUVSOffset:
                startOffset = nonDefUVSOffset - 10
                numRecs, = struct.unpack('>L', data[startOffset:startOffset + 4])
                startOffset += 4
                localUVList = []
                for r in range(numRecs):
                    uv, gid = struct.unpack('>3sH', data[startOffset:startOffset + 5])
                    startOffset += 5
                    uv = cvtToUVS(uv)
                    glyphName = self.ttFont.getGlyphName(gid)
                    localUVList.append((uv, glyphName))
                try:
                    uvsDict[varUVS].extend(localUVList)
                except KeyError:
                    uvsDict[varUVS] = localUVList
        self.uvsDict = uvsDict

    def toXML(self, writer, ttFont):
        writer.begintag(self.__class__.__name__, [('platformID', self.platformID), ('platEncID', self.platEncID)])
        writer.newline()
        uvsDict = self.uvsDict
        uvsList = sorted(uvsDict.keys())
        for uvs in uvsList:
            uvList = uvsDict[uvs]
            uvList.sort(key=lambda item: (item[1] is not None, item[0], item[1]))
            for uv, gname in uvList:
                attrs = [('uv', hex(uv)), ('uvs', hex(uvs))]
                if gname is not None:
                    attrs.append(('name', gname))
                writer.simpletag('map', attrs)
                writer.newline()
        writer.endtag(self.__class__.__name__)
        writer.newline()

    def fromXML(self, name, attrs, content, ttFont):
        self.language = 255
        if not hasattr(self, 'cmap'):
            self.cmap = {}
        if not hasattr(self, 'uvsDict'):
            self.uvsDict = {}
            uvsDict = self.uvsDict
        _hasGlyphNamedNone = None
        for element in content:
            if not isinstance(element, tuple):
                continue
            name, attrs, content = element
            if name != 'map':
                continue
            uvs = safeEval(attrs['uvs'])
            uv = safeEval(attrs['uv'])
            gname = attrs.get('name')
            if gname == 'None':
                if _hasGlyphNamedNone is None:
                    _hasGlyphNamedNone = 'None' in ttFont.getGlyphOrder()
                if not _hasGlyphNamedNone:
                    gname = None
            try:
                uvsDict[uvs].append((uv, gname))
            except KeyError:
                uvsDict[uvs] = [(uv, gname)]

    def compile(self, ttFont):
        if self.data:
            return struct.pack('>HLL', self.format, self.length, self.numVarSelectorRecords) + self.data
        uvsDict = self.uvsDict
        uvsList = sorted(uvsDict.keys())
        self.numVarSelectorRecords = len(uvsList)
        offset = 10 + self.numVarSelectorRecords * 11
        data = []
        varSelectorRecords = []
        for uvs in uvsList:
            entryList = uvsDict[uvs]
            defList = [entry for entry in entryList if entry[1] is None]
            if defList:
                defList = [entry[0] for entry in defList]
                defOVSOffset = offset
                defList.sort()
                lastUV = defList[0]
                cnt = -1
                defRecs = []
                for defEntry in defList:
                    cnt += 1
                    if lastUV + cnt != defEntry:
                        rec = struct.pack('>3sB', cvtFromUVS(lastUV), cnt - 1)
                        lastUV = defEntry
                        defRecs.append(rec)
                        cnt = 0
                rec = struct.pack('>3sB', cvtFromUVS(lastUV), cnt)
                defRecs.append(rec)
                numDefRecs = len(defRecs)
                data.append(struct.pack('>L', numDefRecs))
                data.extend(defRecs)
                offset += 4 + numDefRecs * 4
            else:
                defOVSOffset = 0
            ndefList = [entry for entry in entryList if entry[1] is not None]
            if ndefList:
                nonDefUVSOffset = offset
                ndefList.sort()
                numNonDefRecs = len(ndefList)
                data.append(struct.pack('>L', numNonDefRecs))
                offset += 4 + numNonDefRecs * 5
                for uv, gname in ndefList:
                    gid = ttFont.getGlyphID(gname)
                    ndrec = struct.pack('>3sH', cvtFromUVS(uv), gid)
                    data.append(ndrec)
            else:
                nonDefUVSOffset = 0
            vrec = struct.pack('>3sLL', cvtFromUVS(uvs), defOVSOffset, nonDefUVSOffset)
            varSelectorRecords.append(vrec)
        data = bytesjoin(varSelectorRecords) + bytesjoin(data)
        self.length = 10 + len(data)
        headerdata = struct.pack('>HLL', self.format, self.length, self.numVarSelectorRecords)
        return headerdata + data