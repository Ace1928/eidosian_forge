from fontTools.misc.textTools import bytesjoin, safeEval, readHex
from fontTools.misc.encodingTools import getEncoding
from fontTools.ttLib import getSearchRange
from fontTools.unicode import Unicode
from . import DefaultTable
import sys
import struct
import array
import logging
class cmap_format_0(CmapSubtable):

    def decompile(self, data, ttFont):
        if data is not None and ttFont is not None:
            self.decompileHeader(data, ttFont)
        else:
            assert data is None and ttFont is None, 'Need both data and ttFont arguments'
        data = self.data
        assert 262 == self.length, 'Format 0 cmap subtable not 262 bytes'
        gids = array.array('B')
        gids.frombytes(self.data)
        charCodes = list(range(len(gids)))
        self.cmap = _make_map(self.ttFont, charCodes, gids)

    def compile(self, ttFont):
        if self.data:
            return struct.pack('>HHH', 0, 262, self.language) + self.data
        cmap = self.cmap
        assert set(cmap.keys()).issubset(range(256))
        getGlyphID = ttFont.getGlyphID
        valueList = [getGlyphID(cmap[i]) if i in cmap else 0 for i in range(256)]
        gids = array.array('B', valueList)
        data = struct.pack('>HHH', 0, 262, self.language) + gids.tobytes()
        assert len(data) == 262
        return data

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