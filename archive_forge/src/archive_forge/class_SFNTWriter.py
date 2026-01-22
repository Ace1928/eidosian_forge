from io import BytesIO
from types import SimpleNamespace
from fontTools.misc.textTools import Tag
from fontTools.misc import sstruct
from fontTools.ttLib import TTLibError, TTLibFileIsCollectionError
import struct
from collections import OrderedDict
import logging
class SFNTWriter(object):

    def __new__(cls, *args, **kwargs):
        """Return an instance of the SFNTWriter sub-class which is compatible
        with the specified 'flavor'.
        """
        flavor = None
        if kwargs and 'flavor' in kwargs:
            flavor = kwargs['flavor']
        elif args and len(args) > 3:
            flavor = args[3]
        if cls is SFNTWriter:
            if flavor == 'woff2':
                from fontTools.ttLib.woff2 import WOFF2Writer
                return object.__new__(WOFF2Writer)
        return object.__new__(cls)

    def __init__(self, file, numTables, sfntVersion='\x00\x01\x00\x00', flavor=None, flavorData=None):
        self.file = file
        self.numTables = numTables
        self.sfntVersion = Tag(sfntVersion)
        self.flavor = flavor
        self.flavorData = flavorData
        if self.flavor == 'woff':
            self.directoryFormat = woffDirectoryFormat
            self.directorySize = woffDirectorySize
            self.DirectoryEntry = WOFFDirectoryEntry
            self.signature = 'wOFF'
            self.origNextTableOffset = sfntDirectorySize + numTables * sfntDirectoryEntrySize
        else:
            assert not self.flavor, "Unknown flavor '%s'" % self.flavor
            self.directoryFormat = sfntDirectoryFormat
            self.directorySize = sfntDirectorySize
            self.DirectoryEntry = SFNTDirectoryEntry
            from fontTools.ttLib import getSearchRange
            self.searchRange, self.entrySelector, self.rangeShift = getSearchRange(numTables, 16)
        self.directoryOffset = self.file.tell()
        self.nextTableOffset = self.directoryOffset + self.directorySize + numTables * self.DirectoryEntry.formatSize
        self.file.seek(self.nextTableOffset)
        self.file.write(b'\x00' * (self.nextTableOffset - self.file.tell()))
        self.tables = OrderedDict()

    def setEntry(self, tag, entry):
        if tag in self.tables:
            raise TTLibError("cannot rewrite '%s' table" % tag)
        self.tables[tag] = entry

    def __setitem__(self, tag, data):
        """Write raw table data to disk."""
        if tag in self.tables:
            raise TTLibError("cannot rewrite '%s' table" % tag)
        entry = self.DirectoryEntry()
        entry.tag = tag
        entry.offset = self.nextTableOffset
        if tag == 'head':
            entry.checkSum = calcChecksum(data[:8] + b'\x00\x00\x00\x00' + data[12:])
            self.headTable = data
            entry.uncompressed = True
        else:
            entry.checkSum = calcChecksum(data)
        entry.saveData(self.file, data)
        if self.flavor == 'woff':
            entry.origOffset = self.origNextTableOffset
            self.origNextTableOffset += entry.origLength + 3 & ~3
        self.nextTableOffset = self.nextTableOffset + (entry.length + 3 & ~3)
        self.file.write(b'\x00' * (self.nextTableOffset - self.file.tell()))
        assert self.nextTableOffset == self.file.tell()
        self.setEntry(tag, entry)

    def __getitem__(self, tag):
        return self.tables[tag]

    def close(self):
        """All tables must have been written to disk. Now write the
        directory.
        """
        tables = sorted(self.tables.items())
        if len(tables) != self.numTables:
            raise TTLibError('wrong number of tables; expected %d, found %d' % (self.numTables, len(tables)))
        if self.flavor == 'woff':
            self.signature = b'wOFF'
            self.reserved = 0
            self.totalSfntSize = 12
            self.totalSfntSize += 16 * len(tables)
            for tag, entry in tables:
                self.totalSfntSize += entry.origLength + 3 & ~3
            data = self.flavorData if self.flavorData else WOFFFlavorData()
            if data.majorVersion is not None and data.minorVersion is not None:
                self.majorVersion = data.majorVersion
                self.minorVersion = data.minorVersion
            elif hasattr(self, 'headTable'):
                self.majorVersion, self.minorVersion = struct.unpack('>HH', self.headTable[4:8])
            else:
                self.majorVersion = self.minorVersion = 0
            if data.metaData:
                self.metaOrigLength = len(data.metaData)
                self.file.seek(0, 2)
                self.metaOffset = self.file.tell()
                compressedMetaData = compress(data.metaData)
                self.metaLength = len(compressedMetaData)
                self.file.write(compressedMetaData)
            else:
                self.metaOffset = self.metaLength = self.metaOrigLength = 0
            if data.privData:
                self.file.seek(0, 2)
                off = self.file.tell()
                paddedOff = off + 3 & ~3
                self.file.write(b'\x00' * (paddedOff - off))
                self.privOffset = self.file.tell()
                self.privLength = len(data.privData)
                self.file.write(data.privData)
            else:
                self.privOffset = self.privLength = 0
            self.file.seek(0, 2)
            self.length = self.file.tell()
        else:
            assert not self.flavor, "Unknown flavor '%s'" % self.flavor
            pass
        directory = sstruct.pack(self.directoryFormat, self)
        self.file.seek(self.directoryOffset + self.directorySize)
        seenHead = 0
        for tag, entry in tables:
            if tag == 'head':
                seenHead = 1
            directory = directory + entry.toString()
        if seenHead:
            self.writeMasterChecksum(directory)
        self.file.seek(self.directoryOffset)
        self.file.write(directory)

    def _calcMasterChecksum(self, directory):
        tags = list(self.tables.keys())
        checksums = []
        for i in range(len(tags)):
            checksums.append(self.tables[tags[i]].checkSum)
        if self.DirectoryEntry != SFNTDirectoryEntry:
            from fontTools.ttLib import getSearchRange
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

    def writeMasterChecksum(self, directory):
        checksumadjustment = self._calcMasterChecksum(directory)
        self.file.seek(self.tables['head'].offset + 8)
        self.file.write(struct.pack('>L', checksumadjustment))

    def reordersTables(self):
        return False