from fontTools.misc.textTools import bytesjoin, safeEval, readHex
from fontTools.misc.encodingTools import getEncoding
from fontTools.ttLib import getSearchRange
from fontTools.unicode import Unicode
from . import DefaultTable
import sys
import struct
import array
import logging
class cmap_format_6(CmapSubtable):

    def decompile(self, data, ttFont):
        if data is not None and ttFont is not None:
            self.decompileHeader(data, ttFont)
        else:
            assert data is None and ttFont is None, 'Need both data and ttFont arguments'
        data = self.data
        firstCode, entryCount = struct.unpack('>HH', data[:4])
        firstCode = int(firstCode)
        data = data[4:]
        gids = array.array('H')
        gids.frombytes(data[:2 * int(entryCount)])
        if sys.byteorder != 'big':
            gids.byteswap()
        self.data = data = None
        charCodes = list(range(firstCode, firstCode + len(gids)))
        self.cmap = _make_map(self.ttFont, charCodes, gids)

    def compile(self, ttFont):
        if self.data:
            return struct.pack('>HHH', self.format, self.length, self.language) + self.data
        cmap = self.cmap
        codes = sorted(cmap.keys())
        if codes:
            codes = list(range(codes[0], codes[-1] + 1))
            firstCode = codes[0]
            valueList = [ttFont.getGlyphID(cmap[code]) if code in cmap else 0 for code in codes]
            gids = array.array('H', valueList)
            if sys.byteorder != 'big':
                gids.byteswap()
            data = gids.tobytes()
        else:
            data = b''
            firstCode = 0
        header = struct.pack('>HHHHH', 6, len(data) + 10, self.language, firstCode, len(codes))
        return header + data

    def fromXML(self, name, attrs, content, ttFont):
        self.language = safeEval(attrs['language'])
        if not hasattr(self, 'cmap'):
            self.cmap = {}
        cmap = self.cmap
        for element in content:
            if not isinstance(element, tuple):
                continue
            name, attrs, content = element
            if name != 'map':
                continue
            cmap[safeEval(attrs['code'])] = attrs['name']