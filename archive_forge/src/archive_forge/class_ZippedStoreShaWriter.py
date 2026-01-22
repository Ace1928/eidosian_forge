from io import BytesIO
import mmap
import os
import sys
import zlib
from gitdb.fun import (
from gitdb.util import (
from gitdb.const import NULL_BYTE, BYTE_SPACE
from gitdb.utils.encoding import force_bytes
class ZippedStoreShaWriter(Sha1Writer):
    """Remembers everything someone writes to it and generates a sha"""
    __slots__ = ('buf', 'zip')

    def __init__(self):
        Sha1Writer.__init__(self)
        self.buf = BytesIO()
        self.zip = zlib.compressobj(zlib.Z_BEST_SPEED)

    def __getattr__(self, attr):
        return getattr(self.buf, attr)

    def write(self, data):
        alen = Sha1Writer.write(self, data)
        self.buf.write(self.zip.compress(data))
        return alen

    def close(self):
        self.buf.write(self.zip.flush())

    def seek(self, offset, whence=getattr(os, 'SEEK_SET', 0)):
        """Seeking currently only supports to rewind written data
        Multiple writes are not supported"""
        if offset != 0 or whence != getattr(os, 'SEEK_SET', 0):
            raise ValueError('Can only seek to position 0')
        self.buf.seek(0)

    def getvalue(self):
        """:return: string value from the current stream position to the end"""
        return self.buf.getvalue()