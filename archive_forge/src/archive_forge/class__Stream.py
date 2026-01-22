from builtins import open as bltn_open
import sys
import os
import io
import shutil
import stat
import time
import struct
import copy
import re
import warnings
class _Stream:
    """Class that serves as an adapter between TarFile and
       a stream-like object.  The stream-like object only
       needs to have a read() or write() method that works with bytes,
       and the method is accessed blockwise.
       Use of gzip or bzip2 compression is possible.
       A stream-like object could be for example: sys.stdin.buffer,
       sys.stdout.buffer, a socket, a tape device etc.

       _Stream is intended to be used only internally.
    """

    def __init__(self, name, mode, comptype, fileobj, bufsize):
        """Construct a _Stream object.
        """
        self._extfileobj = True
        if fileobj is None:
            fileobj = _LowLevelFile(name, mode)
            self._extfileobj = False
        if comptype == '*':
            fileobj = _StreamProxy(fileobj)
            comptype = fileobj.getcomptype()
        self.name = name or ''
        self.mode = mode
        self.comptype = comptype
        self.fileobj = fileobj
        self.bufsize = bufsize
        self.buf = b''
        self.pos = 0
        self.closed = False
        try:
            if comptype == 'gz':
                try:
                    import zlib
                except ImportError:
                    raise CompressionError('zlib module is not available') from None
                self.zlib = zlib
                self.crc = zlib.crc32(b'')
                if mode == 'r':
                    self.exception = zlib.error
                    self._init_read_gz()
                else:
                    self._init_write_gz()
            elif comptype == 'bz2':
                try:
                    import bz2
                except ImportError:
                    raise CompressionError('bz2 module is not available') from None
                if mode == 'r':
                    self.dbuf = b''
                    self.cmp = bz2.BZ2Decompressor()
                    self.exception = OSError
                else:
                    self.cmp = bz2.BZ2Compressor()
            elif comptype == 'xz':
                try:
                    import lzma
                except ImportError:
                    raise CompressionError('lzma module is not available') from None
                if mode == 'r':
                    self.dbuf = b''
                    self.cmp = lzma.LZMADecompressor()
                    self.exception = lzma.LZMAError
                else:
                    self.cmp = lzma.LZMACompressor()
            elif comptype != 'tar':
                raise CompressionError('unknown compression type %r' % comptype)
        except:
            if not self._extfileobj:
                self.fileobj.close()
            self.closed = True
            raise

    def __del__(self):
        if hasattr(self, 'closed') and (not self.closed):
            self.close()

    def _init_write_gz(self):
        """Initialize for writing with gzip compression.
        """
        self.cmp = self.zlib.compressobj(9, self.zlib.DEFLATED, -self.zlib.MAX_WBITS, self.zlib.DEF_MEM_LEVEL, 0)
        timestamp = struct.pack('<L', int(time.time()))
        self.__write(b'\x1f\x8b\x08\x08' + timestamp + b'\x02\xff')
        if self.name.endswith('.gz'):
            self.name = self.name[:-3]
        self.name = os.path.basename(self.name)
        self.__write(self.name.encode('iso-8859-1', 'replace') + NUL)

    def write(self, s):
        """Write string s to the stream.
        """
        if self.comptype == 'gz':
            self.crc = self.zlib.crc32(s, self.crc)
        self.pos += len(s)
        if self.comptype != 'tar':
            s = self.cmp.compress(s)
        self.__write(s)

    def __write(self, s):
        """Write string s to the stream if a whole new block
           is ready to be written.
        """
        self.buf += s
        while len(self.buf) > self.bufsize:
            self.fileobj.write(self.buf[:self.bufsize])
            self.buf = self.buf[self.bufsize:]

    def close(self):
        """Close the _Stream object. No operation should be
           done on it afterwards.
        """
        if self.closed:
            return
        self.closed = True
        try:
            if self.mode == 'w' and self.comptype != 'tar':
                self.buf += self.cmp.flush()
            if self.mode == 'w' and self.buf:
                self.fileobj.write(self.buf)
                self.buf = b''
                if self.comptype == 'gz':
                    self.fileobj.write(struct.pack('<L', self.crc))
                    self.fileobj.write(struct.pack('<L', self.pos & 4294967295))
        finally:
            if not self._extfileobj:
                self.fileobj.close()

    def _init_read_gz(self):
        """Initialize for reading a gzip compressed fileobj.
        """
        self.cmp = self.zlib.decompressobj(-self.zlib.MAX_WBITS)
        self.dbuf = b''
        if self.__read(2) != b'\x1f\x8b':
            raise ReadError('not a gzip file')
        if self.__read(1) != b'\x08':
            raise CompressionError('unsupported compression method')
        flag = ord(self.__read(1))
        self.__read(6)
        if flag & 4:
            xlen = ord(self.__read(1)) + 256 * ord(self.__read(1))
            self.read(xlen)
        if flag & 8:
            while True:
                s = self.__read(1)
                if not s or s == NUL:
                    break
        if flag & 16:
            while True:
                s = self.__read(1)
                if not s or s == NUL:
                    break
        if flag & 2:
            self.__read(2)

    def tell(self):
        """Return the stream's file pointer position.
        """
        return self.pos

    def seek(self, pos=0):
        """Set the stream's file pointer to pos. Negative seeking
           is forbidden.
        """
        if pos - self.pos >= 0:
            blocks, remainder = divmod(pos - self.pos, self.bufsize)
            for i in range(blocks):
                self.read(self.bufsize)
            self.read(remainder)
        else:
            raise StreamError('seeking backwards is not allowed')
        return self.pos

    def read(self, size):
        """Return the next size number of bytes from the stream."""
        assert size is not None
        buf = self._read(size)
        self.pos += len(buf)
        return buf

    def _read(self, size):
        """Return size bytes from the stream.
        """
        if self.comptype == 'tar':
            return self.__read(size)
        c = len(self.dbuf)
        t = [self.dbuf]
        while c < size:
            if self.buf:
                buf = self.buf
                self.buf = b''
            else:
                buf = self.fileobj.read(self.bufsize)
                if not buf:
                    break
            try:
                buf = self.cmp.decompress(buf)
            except self.exception as e:
                raise ReadError('invalid compressed data') from e
            t.append(buf)
            c += len(buf)
        t = b''.join(t)
        self.dbuf = t[size:]
        return t[:size]

    def __read(self, size):
        """Return size bytes from stream. If internal buffer is empty,
           read another block from the stream.
        """
        c = len(self.buf)
        t = [self.buf]
        while c < size:
            buf = self.fileobj.read(self.bufsize)
            if not buf:
                break
            t.append(buf)
            c += len(buf)
        t = b''.join(t)
        self.buf = t[size:]
        return t[:size]