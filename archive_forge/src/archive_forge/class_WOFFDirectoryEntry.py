from io import BytesIO
from types import SimpleNamespace
from fontTools.misc.textTools import Tag
from fontTools.misc import sstruct
from fontTools.ttLib import TTLibError, TTLibFileIsCollectionError
import struct
from collections import OrderedDict
import logging
class WOFFDirectoryEntry(DirectoryEntry):
    format = woffDirectoryEntryFormat
    formatSize = woffDirectoryEntrySize

    def __init__(self):
        super(WOFFDirectoryEntry, self).__init__()
        if not hasattr(WOFFDirectoryEntry, 'zlibCompressionLevel'):
            self.zlibCompressionLevel = ZLIB_COMPRESSION_LEVEL

    def decodeData(self, rawData):
        import zlib
        if self.length == self.origLength:
            data = rawData
        else:
            assert self.length < self.origLength
            data = zlib.decompress(rawData)
            assert len(data) == self.origLength
        return data

    def encodeData(self, data):
        self.origLength = len(data)
        if not self.uncompressed:
            compressedData = compress(data, self.zlibCompressionLevel)
        if self.uncompressed or len(compressedData) >= self.origLength:
            rawData = data
            self.length = self.origLength
        else:
            rawData = compressedData
            self.length = len(rawData)
        return rawData