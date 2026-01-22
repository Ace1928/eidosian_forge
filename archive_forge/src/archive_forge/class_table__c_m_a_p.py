from fontTools.misc.textTools import bytesjoin, safeEval, readHex
from fontTools.misc.encodingTools import getEncoding
from fontTools.ttLib import getSearchRange
from fontTools.unicode import Unicode
from . import DefaultTable
import sys
import struct
import array
import logging
class table__c_m_a_p(DefaultTable.DefaultTable):
    """Character to Glyph Index Mapping Table

    This class represents the `cmap <https://docs.microsoft.com/en-us/typography/opentype/spec/cmap>`_
    table, which maps between input characters (in Unicode or other system encodings)
    and glyphs within the font. The ``cmap`` table contains one or more subtables
    which determine the mapping of of characters to glyphs across different platforms
    and encoding systems.

    ``table__c_m_a_p`` objects expose an accessor ``.tables`` which provides access
    to the subtables, although it is normally easier to retrieve individual subtables
    through the utility methods described below. To add new subtables to a font,
    first determine the subtable format (if in doubt use format 4 for glyphs within
    the BMP, format 12 for glyphs outside the BMP, and format 14 for Unicode Variation
    Sequences) construct subtable objects with ``CmapSubtable.newSubtable(format)``,
    and append them to the ``.tables`` list.

    Within a subtable, the mapping of characters to glyphs is provided by the ``.cmap``
    attribute.

    Example::

            cmap4_0_3 = CmapSubtable.newSubtable(4)
            cmap4_0_3.platformID = 0
            cmap4_0_3.platEncID = 3
            cmap4_0_3.language = 0
            cmap4_0_3.cmap = { 0xC1: "Aacute" }

            cmap = newTable("cmap")
            cmap.tableVersion = 0
            cmap.tables = [cmap4_0_3]
    """

    def getcmap(self, platformID, platEncID):
        """Returns the first subtable which matches the given platform and encoding.

        Args:
                platformID (int): The platform ID. Use 0 for Unicode, 1 for Macintosh
                        (deprecated for new fonts), 2 for ISO (deprecated) and 3 for Windows.
                encodingID (int): Encoding ID. Interpretation depends on the platform ID.
                        See the OpenType specification for details.

        Returns:
                An object which is a subclass of :py:class:`CmapSubtable` if a matching
                subtable is found within the font, or ``None`` otherwise.
        """
        for subtable in self.tables:
            if subtable.platformID == platformID and subtable.platEncID == platEncID:
                return subtable
        return None

    def getBestCmap(self, cmapPreferences=((3, 10), (0, 6), (0, 4), (3, 1), (0, 3), (0, 2), (0, 1), (0, 0))):
        """Returns the 'best' Unicode cmap dictionary available in the font
        or ``None``, if no Unicode cmap subtable is available.

        By default it will search for the following (platformID, platEncID)
        pairs in order::

                        (3, 10), # Windows Unicode full repertoire
                        (0, 6),  # Unicode full repertoire (format 13 subtable)
                        (0, 4),  # Unicode 2.0 full repertoire
                        (3, 1),  # Windows Unicode BMP
                        (0, 3),  # Unicode 2.0 BMP
                        (0, 2),  # Unicode ISO/IEC 10646
                        (0, 1),  # Unicode 1.1
                        (0, 0)   # Unicode 1.0

        This particular order matches what HarfBuzz uses to choose what
        subtable to use by default. This order prefers the largest-repertoire
        subtable, and among those, prefers the Windows-platform over the
        Unicode-platform as the former has wider support.

        This order can be customized via the ``cmapPreferences`` argument.
        """
        for platformID, platEncID in cmapPreferences:
            cmapSubtable = self.getcmap(platformID, platEncID)
            if cmapSubtable is not None:
                return cmapSubtable.cmap
        return None

    def buildReversed(self):
        """Builds a reverse mapping dictionary

        Iterates over all Unicode cmap tables and returns a dictionary mapping
        glyphs to sets of codepoints, such as::

                {
                        'one': {0x31}
                        'A': {0x41,0x391}
                }

        The values are sets of Unicode codepoints because
        some fonts map different codepoints to the same glyph.
        For example, ``U+0041 LATIN CAPITAL LETTER A`` and ``U+0391
        GREEK CAPITAL LETTER ALPHA`` are sometimes the same glyph.
        """
        result = {}
        for subtable in self.tables:
            if subtable.isUnicode():
                for codepoint, name in subtable.cmap.items():
                    result.setdefault(name, set()).add(codepoint)
        return result

    def decompile(self, data, ttFont):
        tableVersion, numSubTables = struct.unpack('>HH', data[:4])
        self.tableVersion = int(tableVersion)
        self.tables = tables = []
        seenOffsets = {}
        for i in range(numSubTables):
            platformID, platEncID, offset = struct.unpack('>HHl', data[4 + i * 8:4 + (i + 1) * 8])
            platformID, platEncID = (int(platformID), int(platEncID))
            format, length = struct.unpack('>HH', data[offset:offset + 4])
            if format in [8, 10, 12, 13]:
                format, reserved, length = struct.unpack('>HHL', data[offset:offset + 8])
            elif format in [14]:
                format, length = struct.unpack('>HL', data[offset:offset + 6])
            if not length:
                log.error('cmap subtable is reported as having zero length: platformID %s, platEncID %s, format %s offset %s. Skipping table.', platformID, platEncID, format, offset)
                continue
            table = CmapSubtable.newSubtable(format)
            table.platformID = platformID
            table.platEncID = platEncID
            table.decompileHeader(data[offset:offset + int(length)], ttFont)
            if offset in seenOffsets:
                table.data = None
                table.cmap = tables[seenOffsets[offset]].cmap
            else:
                seenOffsets[offset] = i
            tables.append(table)
        if ttFont.lazy is False:
            self.ensureDecompiled()

    def ensureDecompiled(self, recurse=False):
        for st in self.tables:
            st.ensureDecompiled()

    def compile(self, ttFont):
        self.tables.sort()
        numSubTables = len(self.tables)
        totalOffset = 4 + 8 * numSubTables
        data = struct.pack('>HH', self.tableVersion, numSubTables)
        tableData = b''
        seen = {}
        done = {}
        for table in self.tables:
            offset = seen.get(id(table.cmap))
            if offset is None:
                chunk = table.compile(ttFont)
                offset = done.get(chunk)
                if offset is None:
                    offset = seen[id(table.cmap)] = done[chunk] = totalOffset + len(tableData)
                    tableData = tableData + chunk
            data = data + struct.pack('>HHl', table.platformID, table.platEncID, offset)
        return data + tableData

    def toXML(self, writer, ttFont):
        writer.simpletag('tableVersion', version=self.tableVersion)
        writer.newline()
        for table in self.tables:
            table.toXML(writer, ttFont)

    def fromXML(self, name, attrs, content, ttFont):
        if name == 'tableVersion':
            self.tableVersion = safeEval(attrs['version'])
            return
        if name[:12] != 'cmap_format_':
            return
        if not hasattr(self, 'tables'):
            self.tables = []
        format = safeEval(name[12:])
        table = CmapSubtable.newSubtable(format)
        table.platformID = safeEval(attrs['platformID'])
        table.platEncID = safeEval(attrs['platEncID'])
        table.fromXML(name, attrs, content, ttFont)
        self.tables.append(table)