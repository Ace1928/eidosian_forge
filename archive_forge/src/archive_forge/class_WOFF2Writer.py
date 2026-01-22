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
class WOFF2Writer(SFNTWriter):
    flavor = 'woff2'

    def __init__(self, file, numTables, sfntVersion='\x00\x01\x00\x00', flavor=None, flavorData=None):
        if not haveBrotli:
            log.error('The WOFF2 encoder requires the Brotli Python extension, available at: https://github.com/google/brotli')
            raise ImportError('No module named brotli')
        self.file = file
        self.numTables = numTables
        self.sfntVersion = Tag(sfntVersion)
        self.flavorData = WOFF2FlavorData(data=flavorData)
        self.directoryFormat = woff2DirectoryFormat
        self.directorySize = woff2DirectorySize
        self.DirectoryEntry = WOFF2DirectoryEntry
        self.signature = Tag('wOF2')
        self.nextTableOffset = 0
        self.transformBuffer = BytesIO()
        self.tables = OrderedDict()
        self.ttFont = TTFont(recalcBBoxes=False, recalcTimestamp=False)

    def __setitem__(self, tag, data):
        """Associate new entry named 'tag' with raw table data."""
        if tag in self.tables:
            raise TTLibError("cannot rewrite '%s' table" % tag)
        if tag == 'DSIG':
            self.numTables -= 1
            return
        entry = self.DirectoryEntry()
        entry.tag = Tag(tag)
        entry.flags = getKnownTagIndex(entry.tag)
        entry.data = data
        self.tables[tag] = entry

    def close(self):
        """All tags must have been specified. Now write the table data and directory."""
        if len(self.tables) != self.numTables:
            raise TTLibError('wrong number of tables; expected %d, found %d' % (self.numTables, len(self.tables)))
        if self.sfntVersion in ('\x00\x01\x00\x00', 'true'):
            isTrueType = True
        elif self.sfntVersion == 'OTTO':
            isTrueType = False
        else:
            raise TTLibError('Not a TrueType or OpenType font (bad sfntVersion)')
        if isTrueType and 'glyf' in self.flavorData.transformedTables and ('glyf' in self.tables):
            self._normaliseGlyfAndLoca(padding=4)
        self._setHeadTransformFlag()
        self.tables = OrderedDict(sorted(self.tables.items()))
        self.totalSfntSize = self._calcSFNTChecksumsLengthsAndOffsets()
        fontData = self._transformTables()
        compressedFont = brotli.compress(fontData, mode=brotli.MODE_FONT)
        self.totalCompressedSize = len(compressedFont)
        self.length = self._calcTotalSize()
        self.majorVersion, self.minorVersion = self._getVersion()
        self.reserved = 0
        directory = self._packTableDirectory()
        self.file.seek(0)
        self.file.write(pad(directory + compressedFont, size=4))
        self._writeFlavorData()

    def _normaliseGlyfAndLoca(self, padding=4):
        """Recompile glyf and loca tables, aligning glyph offsets to multiples of
        'padding' size. Update the head table's 'indexToLocFormat' accordingly while
        compiling loca.
        """
        if self.sfntVersion == 'OTTO':
            return
        for tag in ('maxp', 'head', 'loca', 'glyf', 'fvar'):
            if tag in self.tables:
                self._decompileTable(tag)
        self.ttFont['glyf'].padding = padding
        for tag in ('glyf', 'loca'):
            self._compileTable(tag)

    def _setHeadTransformFlag(self):
        """Set bit 11 of 'head' table flags to indicate that the font has undergone
        a lossless modifying transform. Re-compile head table data."""
        self._decompileTable('head')
        self.ttFont['head'].flags |= 1 << 11
        self._compileTable('head')

    def _decompileTable(self, tag):
        """Fetch table data, decompile it, and store it inside self.ttFont."""
        tag = Tag(tag)
        if tag not in self.tables:
            raise TTLibError('missing required table: %s' % tag)
        if self.ttFont.isLoaded(tag):
            return
        data = self.tables[tag].data
        if tag == 'loca':
            tableClass = WOFF2LocaTable
        elif tag == 'glyf':
            tableClass = WOFF2GlyfTable
        elif tag == 'hmtx':
            tableClass = WOFF2HmtxTable
        else:
            tableClass = getTableClass(tag)
        table = tableClass(tag)
        self.ttFont.tables[tag] = table
        table.decompile(data, self.ttFont)

    def _compileTable(self, tag):
        """Compile table and store it in its 'data' attribute."""
        self.tables[tag].data = self.ttFont[tag].compile(self.ttFont)

    def _calcSFNTChecksumsLengthsAndOffsets(self):
        """Compute the 'original' SFNT checksums, lengths and offsets for checksum
        adjustment calculation. Return the total size of the uncompressed font.
        """
        offset = sfntDirectorySize + sfntDirectoryEntrySize * len(self.tables)
        for tag, entry in self.tables.items():
            data = entry.data
            entry.origOffset = offset
            entry.origLength = len(data)
            if tag == 'head':
                entry.checkSum = calcChecksum(data[:8] + b'\x00\x00\x00\x00' + data[12:])
            else:
                entry.checkSum = calcChecksum(data)
            offset += entry.origLength + 3 & ~3
        return offset

    def _transformTables(self):
        """Return transformed font data."""
        transformedTables = self.flavorData.transformedTables
        for tag, entry in self.tables.items():
            data = None
            if tag in transformedTables:
                data = self.transformTable(tag)
                if data is not None:
                    entry.transformed = True
            if data is None:
                if tag == 'glyf':
                    transformedTables.discard('loca')
                data = entry.data
                entry.transformed = False
            entry.offset = self.nextTableOffset
            entry.saveData(self.transformBuffer, data)
            self.nextTableOffset += entry.length
        self.writeMasterChecksum()
        fontData = self.transformBuffer.getvalue()
        return fontData

    def transformTable(self, tag):
        """Return transformed table data, or None if some pre-conditions aren't
        met -- in which case, the non-transformed table data will be used.
        """
        if tag == 'loca':
            data = b''
        elif tag == 'glyf':
            for tag in ('maxp', 'head', 'loca', 'glyf'):
                self._decompileTable(tag)
            glyfTable = self.ttFont['glyf']
            data = glyfTable.transform(self.ttFont)
        elif tag == 'hmtx':
            if 'glyf' not in self.tables:
                return
            for tag in ('maxp', 'head', 'hhea', 'loca', 'glyf', 'hmtx'):
                self._decompileTable(tag)
            hmtxTable = self.ttFont['hmtx']
            data = hmtxTable.transform(self.ttFont)
        else:
            raise TTLibError("Transform for table '%s' is unknown" % tag)
        return data

    def _calcMasterChecksum(self):
        """Calculate checkSumAdjustment."""
        tags = list(self.tables.keys())
        checksums = []
        for i in range(len(tags)):
            checksums.append(self.tables[tags[i]].checkSum)
        self.searchRange, self.entrySelector, self.rangeShift = getSearchRange(self.numTables, 16)
        directory = sstruct.pack(sfntDirectoryFormat, self)
        tables = sorted(self.tables.items())
        for tag, entry in tables:
            sfntEntry = SFNTDirectoryEntry()
            sfntEntry.tag = entry.tag
            sfntEntry.checkSum = entry.checkSum
            sfntEntry.offset = entry.origOffset
            sfntEntry.length = entry.origLength
            directory = directory + sfntEntry.toString()
        directory_end = sfntDirectorySize + len(self.tables) * sfntDirectoryEntrySize
        assert directory_end == len(directory)
        checksums.append(calcChecksum(directory))
        checksum = sum(checksums) & 4294967295
        checksumadjustment = 2981146554 - checksum & 4294967295
        return checksumadjustment

    def writeMasterChecksum(self):
        """Write checkSumAdjustment to the transformBuffer."""
        checksumadjustment = self._calcMasterChecksum()
        self.transformBuffer.seek(self.tables['head'].offset + 8)
        self.transformBuffer.write(struct.pack('>L', checksumadjustment))

    def _calcTotalSize(self):
        """Calculate total size of WOFF2 font, including any meta- and/or private data."""
        offset = self.directorySize
        for entry in self.tables.values():
            offset += len(entry.toString())
        offset += self.totalCompressedSize
        offset = offset + 3 & ~3
        offset = self._calcFlavorDataOffsetsAndSize(offset)
        return offset

    def _calcFlavorDataOffsetsAndSize(self, start):
        """Calculate offsets and lengths for any meta- and/or private data."""
        offset = start
        data = self.flavorData
        if data.metaData:
            self.metaOrigLength = len(data.metaData)
            self.metaOffset = offset
            self.compressedMetaData = brotli.compress(data.metaData, mode=brotli.MODE_TEXT)
            self.metaLength = len(self.compressedMetaData)
            offset += self.metaLength
        else:
            self.metaOffset = self.metaLength = self.metaOrigLength = 0
            self.compressedMetaData = b''
        if data.privData:
            offset = offset + 3 & ~3
            self.privOffset = offset
            self.privLength = len(data.privData)
            offset += self.privLength
        else:
            self.privOffset = self.privLength = 0
        return offset

    def _getVersion(self):
        """Return the WOFF2 font's (majorVersion, minorVersion) tuple."""
        data = self.flavorData
        if data.majorVersion is not None and data.minorVersion is not None:
            return (data.majorVersion, data.minorVersion)
        elif 'head' in self.tables:
            return struct.unpack('>HH', self.tables['head'].data[4:8])
        else:
            return (0, 0)

    def _packTableDirectory(self):
        """Return WOFF2 table directory data."""
        directory = sstruct.pack(self.directoryFormat, self)
        for entry in self.tables.values():
            directory = directory + entry.toString()
        return directory

    def _writeFlavorData(self):
        """Write metadata and/or private data using appropiate padding."""
        compressedMetaData = self.compressedMetaData
        privData = self.flavorData.privData
        if compressedMetaData and privData:
            compressedMetaData = pad(compressedMetaData, size=4)
        if compressedMetaData:
            self.file.seek(self.metaOffset)
            assert self.file.tell() == self.metaOffset
            self.file.write(compressedMetaData)
        if privData:
            self.file.seek(self.privOffset)
            assert self.file.tell() == self.privOffset
            self.file.write(privData)

    def reordersTables(self):
        return True