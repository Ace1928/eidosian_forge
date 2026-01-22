from __future__ import absolute_import, unicode_literals
import io
import os
from ._cffi import (  # type: ignore
class ZstdDecompressionWriter(object):
    """
    Write-only stream wrapper that performs decompression.

    This type provides a writable stream that performs decompression and writes
    decompressed data to another stream.

    This type implements the ``io.RawIOBase`` interface. Only methods that
    involve writing will do useful things.

    Behavior is similar to :py:meth:`ZstdCompressor.stream_writer`: compressed
    data is sent to the decompressor by calling ``write(data)`` and decompressed
    output is written to the inner stream by calling its ``write(data)``
    method:

    >>> dctx = zstandard.ZstdDecompressor()
    >>> decompressor = dctx.stream_writer(fh)
    >>> # Will call fh.write() with uncompressed data.
    >>> decompressor.write(compressed_data)

    Instances can be used as context managers. However, context managers add no
    extra special behavior other than automatically calling ``close()`` when
    they exit.

    Calling ``close()`` will mark the stream as closed and subsequent I/O
    operations will raise ``ValueError`` (per the documented behavior of
    ``io.RawIOBase``). ``close()`` will also call ``close()`` on the
    underlying stream if such a method exists and the instance was created with
    ``closefd=True``.

    The size of chunks to ``write()`` to the destination can be specified:

    >>> dctx = zstandard.ZstdDecompressor()
    >>> with dctx.stream_writer(fh, write_size=16384) as decompressor:
    >>>    pass

    You can see how much memory is being used by the decompressor:

    >>> dctx = zstandard.ZstdDecompressor()
    >>> with dctx.stream_writer(fh) as decompressor:
    >>>    byte_size = decompressor.memory_size()

    ``stream_writer()`` accepts a ``write_return_read`` boolean argument to control
    the return value of ``write()``. When ``True`` (the default)``, ``write()``
    returns the number of bytes that were read from the input. When ``False``,
    ``write()`` returns the number of bytes that were ``write()`` to the inner
    stream.
    """

    def __init__(self, decompressor, writer, write_size, write_return_read, closefd=True):
        decompressor._ensure_dctx()
        self._decompressor = decompressor
        self._writer = writer
        self._write_size = write_size
        self._write_return_read = bool(write_return_read)
        self._closefd = bool(closefd)
        self._entered = False
        self._closing = False
        self._closed = False

    def __enter__(self):
        if self._closed:
            raise ValueError('stream is closed')
        if self._entered:
            raise ZstdError('cannot __enter__ multiple times')
        self._entered = True
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self._entered = False
        self.close()
        return False

    def __iter__(self):
        raise io.UnsupportedOperation()

    def __next__(self):
        raise io.UnsupportedOperation()

    def memory_size(self):
        return lib.ZSTD_sizeof_DCtx(self._decompressor._dctx)

    def close(self):
        if self._closed:
            return
        try:
            self._closing = True
            self.flush()
        finally:
            self._closing = False
            self._closed = True
        f = getattr(self._writer, 'close', None)
        if self._closefd and f:
            f()

    @property
    def closed(self):
        return self._closed

    def fileno(self):
        f = getattr(self._writer, 'fileno', None)
        if f:
            return f()
        else:
            raise OSError('fileno not available on underlying writer')

    def flush(self):
        if self._closed:
            raise ValueError('stream is closed')
        f = getattr(self._writer, 'flush', None)
        if f and (not self._closing):
            return f()

    def isatty(self):
        return False

    def readable(self):
        return False

    def readline(self, size=-1):
        raise io.UnsupportedOperation()

    def readlines(self, hint=-1):
        raise io.UnsupportedOperation()

    def seek(self, offset, whence=None):
        raise io.UnsupportedOperation()

    def seekable(self):
        return False

    def tell(self):
        raise io.UnsupportedOperation()

    def truncate(self, size=None):
        raise io.UnsupportedOperation()

    def writable(self):
        return True

    def writelines(self, lines):
        raise io.UnsupportedOperation()

    def read(self, size=-1):
        raise io.UnsupportedOperation()

    def readall(self):
        raise io.UnsupportedOperation()

    def readinto(self, b):
        raise io.UnsupportedOperation()

    def write(self, data):
        if self._closed:
            raise ValueError('stream is closed')
        total_write = 0
        in_buffer = ffi.new('ZSTD_inBuffer *')
        out_buffer = ffi.new('ZSTD_outBuffer *')
        data_buffer = ffi.from_buffer(data)
        in_buffer.src = data_buffer
        in_buffer.size = len(data_buffer)
        in_buffer.pos = 0
        dst_buffer = ffi.new('char[]', self._write_size)
        out_buffer.dst = dst_buffer
        out_buffer.size = len(dst_buffer)
        out_buffer.pos = 0
        dctx = self._decompressor._dctx
        while in_buffer.pos < in_buffer.size:
            zresult = lib.ZSTD_decompressStream(dctx, out_buffer, in_buffer)
            if lib.ZSTD_isError(zresult):
                raise ZstdError('zstd decompress error: %s' % _zstd_error(zresult))
            if out_buffer.pos:
                self._writer.write(ffi.buffer(out_buffer.dst, out_buffer.pos)[:])
                total_write += out_buffer.pos
                out_buffer.pos = 0
        if self._write_return_read:
            return in_buffer.pos
        else:
            return total_write