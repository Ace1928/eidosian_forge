from io import BytesIO
import mmap
import os
import sys
import zlib
from gitdb.fun import (
from gitdb.util import (
from gitdb.const import NULL_BYTE, BYTE_SPACE
from gitdb.utils.encoding import force_bytes
class FDCompressedSha1Writer(Sha1Writer):
    """Digests data written to it, making the sha available, then compress the
    data and write it to the file descriptor

    **Note:** operates on raw file descriptors
    **Note:** for this to work, you have to use the close-method of this instance"""
    __slots__ = ('fd', 'sha1', 'zip')
    exc = IOError('Failed to write all bytes to filedescriptor')

    def __init__(self, fd):
        super().__init__()
        self.fd = fd
        self.zip = zlib.compressobj(zlib.Z_BEST_SPEED)

    def write(self, data):
        """:raise IOError: If not all bytes could be written
        :return: length of incoming data"""
        self.sha1.update(data)
        cdata = self.zip.compress(data)
        bytes_written = write(self.fd, cdata)
        if bytes_written != len(cdata):
            raise self.exc
        return len(data)

    def close(self):
        remainder = self.zip.flush()
        if write(self.fd, remainder) != len(remainder):
            raise self.exc
        return close(self.fd)