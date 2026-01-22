import os.path
import struct
import zipfile
import zlib
class ZipFileEntry(_FileEntry):
    """
    File-like object used to read an uncompressed entry in a ZipFile
    """

    def __init__(self, chunkingZipFile, length):
        _FileEntry.__init__(self, chunkingZipFile, length)
        self.readBytes = 0

    def tell(self):
        return self.readBytes

    def read(self, n=None):
        if n is None:
            n = self.length - self.readBytes
        if n == 0 or self.finished:
            return b''
        data = self.chunkingZipFile.fp.read(min(n, self.length - self.readBytes))
        self.readBytes += len(data)
        if self.readBytes == self.length or len(data) < n:
            self.finished = 1
        return data