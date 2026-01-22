from fontTools.misc.fixedTools import (
from fontTools.misc.roundTools import nearestMultipleShortestRepr, otRound
from fontTools.misc.textTools import bytesjoin, tobytes, tostr, pad, safeEval
from fontTools.ttLib import getSearchRange
from .otBase import (
from .otTables import (
from itertools import zip_longest
from functools import partial
import re
import struct
from typing import Optional
import logging
class GlyphCIDMap(BaseConverter):

    def read(self, reader, font, tableDict):
        glyphOrder = font.getGlyphOrder()
        count = reader.readUShort()
        cids = reader.readUShortArray(count)
        if count > len(glyphOrder):
            log.warning('GlyphCIDMap has %d elements, but the font has only %d glyphs; ignoring the rest' % (count, len(glyphOrder)))
        result = {}
        for glyphID in range(min(len(cids), len(glyphOrder))):
            cid = cids[glyphID]
            if cid != 65535:
                result[glyphOrder[glyphID]] = cid
        return result

    def write(self, writer, font, tableDict, value, repeatIndex=None):
        items = {font.getGlyphID(g): cid for g, cid in value.items() if cid is not None and cid != 65535}
        count = max(items) + 1 if items else 0
        writer.writeUShort(count)
        for glyphID in range(count):
            writer.writeUShort(items.get(glyphID, 65535))

    def xmlRead(self, attrs, content, font):
        result = {}
        for eName, eAttrs, _eContent in filter(istuple, content):
            if eName == 'CID':
                result[eAttrs['glyph']] = safeEval(eAttrs['value'])
        return result

    def xmlWrite(self, xmlWriter, font, value, name, attrs):
        xmlWriter.begintag(name, attrs)
        xmlWriter.newline()
        for glyph, cid in sorted(value.items()):
            if cid is not None and cid != 65535:
                xmlWriter.simpletag('CID', glyph=glyph, value=cid)
                xmlWriter.newline()
        xmlWriter.endtag(name)
        xmlWriter.newline()