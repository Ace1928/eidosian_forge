from io import BytesIO
from types import SimpleNamespace
from fontTools.misc.textTools import Tag
from fontTools.misc import sstruct
from fontTools.ttLib import TTLibError, TTLibFileIsCollectionError
import struct
from collections import OrderedDict
import logging
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