import os
from boto.exception import StorageResponseError
class KeyFile:

    def __init__(self, key):
        self.key = key
        self.key.open_read()
        self.location = 0
        self.closed = False
        self.softspace = -1
        self.mode = 'r'
        self.encoding = 'Undefined in KeyFile'
        self.errors = 'Undefined in KeyFile'
        self.newlines = 'Undefined in KeyFile'
        self.name = key.name

    def tell(self):
        if self.location is None:
            raise ValueError('I/O operation on closed file')
        return self.location

    def seek(self, pos, whence=os.SEEK_SET):
        self.key.close(fast=True)
        if whence == os.SEEK_END:
            if self.key.size == 0:
                return
            pos = self.key.size + pos - 1
            if pos < 0:
                raise IOError('Invalid argument')
            self.key.open_read(headers={'Range': 'bytes=%d-' % pos})
            self.key.read(1)
            self.location = pos + 1
            return
        if whence == os.SEEK_SET:
            if pos < 0:
                raise IOError('Invalid argument')
        elif whence == os.SEEK_CUR:
            pos += self.location
        else:
            raise IOError('Invalid whence param (%d) passed to seek' % whence)
        try:
            self.key.open_read(headers={'Range': 'bytes=%d-' % pos})
        except StorageResponseError as e:
            if e.status != 416:
                raise
        self.location = pos

    def read(self, size):
        self.location += size
        return self.key.read(size)

    def close(self):
        self.key.close()
        self.location = None
        self.closed = True

    def isatty(self):
        return False

    def getkey(self):
        return self.key

    def write(self, buf):
        raise NotImplementedError('write not implemented in KeyFile')

    def fileno(self):
        raise NotImplementedError('fileno not implemented in KeyFile')

    def flush(self):
        raise NotImplementedError('flush not implemented in KeyFile')

    def next(self):
        raise NotImplementedError('next not implemented in KeyFile')

    def readinto(self):
        raise NotImplementedError('readinto not implemented in KeyFile')

    def readline(self):
        raise NotImplementedError('readline not implemented in KeyFile')

    def readlines(self):
        raise NotImplementedError('readlines not implemented in KeyFile')

    def truncate(self):
        raise NotImplementedError('truncate not implemented in KeyFile')

    def writelines(self):
        raise NotImplementedError('writelines not implemented in KeyFile')

    def xreadlines(self):
        raise NotImplementedError('xreadlines not implemented in KeyFile')