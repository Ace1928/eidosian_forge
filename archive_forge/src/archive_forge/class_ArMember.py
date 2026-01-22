from pathlib import Path
import sys
class ArMember(object):
    """ Member of an ar archive.

    Implements most of a file object interface: read, readline, next,
    readlines, seek, tell, close.

    ArMember objects have the following (read-only) properties:
        - name      member name in an ar archive
        - mtime     modification time
        - owner     owner user
        - group     owner group
        - fmode     file permissions
        - size      size in bytes
        - fname     file name"""

    def __init__(self):
        self.__name = None
        self.__mtime = None
        self.__owner = None
        self.__group = None
        self.__fmode = None
        self.__size = None
        self.__fname = ''
        self.__fp = None
        self.__offset = 0
        self.__end = 0
        self.__cur = 0

    @staticmethod
    def from_file(fp, fname, encoding=None, errors=None):
        """fp is an open File object positioned on a valid file header inside
        an ar archive. Return a new ArMember on success, None otherwise. """
        buf = fp.read(FILE_HEADER_LENGTH)
        if not buf:
            return None
        if len(buf) < FILE_HEADER_LENGTH:
            raise IOError('Incorrect header length')
        if buf[58:60] != FILE_MAGIC:
            raise IOError('Incorrect file magic')
        if encoding is None:
            encoding = sys.getfilesystemencoding()
        if errors is None:
            errors = 'surrogateescape'
        f = ArMember()
        name = buf[0:16].split(b'/')[0].strip()
        f.__name = name.decode(encoding, errors)
        f.__mtime = int(buf[16:28])
        f.__owner = int(buf[28:34])
        f.__group = int(buf[34:40])
        f.__fmode = buf[40:48]
        f.__size = int(buf[48:58])
        f.__fname = fname
        if not fname:
            f.__fp = fp
        f.__offset = fp.tell()
        f.__end = f.__offset + f.__size
        f.__cur = f.__offset
        return f

    def read(self, size=0):
        if self.__fp is None:
            if self.__fname is None:
                raise ValueError('Cannot have both fp and fname undefined')
            self.__fp = open(self.__fname, 'rb')
        self.__fp.seek(self.__cur)
        if 0 < size <= self.__end - self.__cur:
            buf = self.__fp.read(size)
            self.__cur = self.__fp.tell()
            return buf
        if self.__cur >= self.__end or self.__cur < self.__offset:
            return b''
        buf = self.__fp.read(self.__end - self.__cur)
        self.__cur = self.__fp.tell()
        return buf

    def readline(self, size=None):
        if self.__fp is None:
            if self.__fname is None:
                raise ValueError('Cannot have both fp and fname undefined')
            self.__fp = open(self.__fname, 'rb')
        self.__fp.seek(self.__cur)
        if size is not None:
            buf = self.__fp.readline(size)
            self.__cur = self.__fp.tell()
            if self.__cur > self.__end:
                return b''
            return buf
        buf = self.__fp.readline()
        self.__cur = self.__fp.tell()
        if self.__cur > self.__end:
            return b''
        return buf

    def readlines(self, sizehint=0):
        buf = None
        lines = []
        while True:
            buf = self.readline()
            if not buf:
                break
            lines.append(buf)
        return lines

    def seek(self, offset, whence=0):
        if self.__cur < self.__offset:
            self.__cur = self.__offset
        if whence < 2 and offset + self.__cur < self.__offset:
            raise IOError("Can't seek at %d" % offset)
        if whence == 1:
            self.__cur = self.__cur + offset
        elif whence == 0:
            self.__cur = self.__offset + offset
        elif whence == 2:
            self.__cur = self.__end + offset

    def tell(self):
        if self.__cur < self.__offset:
            return 0
        return self.__cur - self.__offset

    def seekable(self):
        return True

    def close(self):
        if self.__fp is not None and self.__fname is not None:
            self.__fp.close()
            self.__fp = None

    def next(self):
        return self.readline()

    def __iter__(self):

        def nextline():
            line = self.readline()
            if line:
                yield line
        return iter(nextline())
    name = property(lambda self: self.__name)
    mtime = property(lambda self: self.__mtime)
    owner = property(lambda self: self.__owner)
    group = property(lambda self: self.__group)
    fmode = property(lambda self: self.__fmode)
    size = property(lambda self: self.__size)
    fname = property(lambda self: self.__fname)