from __future__ import absolute_import, unicode_literals
import io
import os
from ._cffi import (  # type: ignore
class ZstdDecompressionReader(object):
    """Read only decompressor that pull uncompressed data from another stream.

    This type provides a read-only stream interface for performing transparent
    decompression from another stream or data source. It conforms to the
    ``io.RawIOBase`` interface. Only methods relevant to reading are
    implemented.

    >>> with open(path, 'rb') as fh:
    >>> dctx = zstandard.ZstdDecompressor()
    >>> reader = dctx.stream_reader(fh)
    >>> while True:
    ...     chunk = reader.read(16384)
    ...     if not chunk:
    ...         break
    ...     # Do something with decompressed chunk.

    The stream can also be used as a context manager:

    >>> with open(path, 'rb') as fh:
    ...     dctx = zstandard.ZstdDecompressor()
    ...     with dctx.stream_reader(fh) as reader:
    ...         ...

    When used as a context manager, the stream is closed and the underlying
    resources are released when the context manager exits. Future operations
    against the stream will fail.

    The ``source`` argument to ``stream_reader()`` can be any object with a
    ``read(size)`` method or any object implementing the *buffer protocol*.

    If the ``source`` is a stream, you can specify how large ``read()`` requests
    to that stream should be via the ``read_size`` argument. It defaults to
    ``zstandard.DECOMPRESSION_RECOMMENDED_INPUT_SIZE``.:

    >>> with open(path, 'rb') as fh:
    ...     dctx = zstandard.ZstdDecompressor()
    ...     # Will perform fh.read(8192) when obtaining data for the decompressor.
    ...     with dctx.stream_reader(fh, read_size=8192) as reader:
    ...         ...

    Instances are *partially* seekable. Absolute and relative positions
    (``SEEK_SET`` and ``SEEK_CUR``) forward of the current position are
    allowed. Offsets behind the current read position and offsets relative
    to the end of stream are not allowed and will raise ``ValueError``
    if attempted.

    ``tell()`` returns the number of decompressed bytes read so far.

    Not all I/O methods are implemented. Notably missing is support for
    ``readline()``, ``readlines()``, and linewise iteration support. This is
    because streams operate on binary data - not text data. If you want to
    convert decompressed output to text, you can chain an ``io.TextIOWrapper``
    to the stream:

    >>> with open(path, 'rb') as fh:
    ...     dctx = zstandard.ZstdDecompressor()
    ...     stream_reader = dctx.stream_reader(fh)
    ...     text_stream = io.TextIOWrapper(stream_reader, encoding='utf-8')
    ...     for line in text_stream:
    ...         ...
    """

    def __init__(self, decompressor, source, read_size, read_across_frames, closefd=True):
        self._decompressor = decompressor
        self._source = source
        self._read_size = read_size
        self._read_across_frames = bool(read_across_frames)
        self._closefd = bool(closefd)
        self._entered = False
        self._closed = False
        self._bytes_decompressed = 0
        self._finished_input = False
        self._finished_output = False
        self._in_buffer = ffi.new('ZSTD_inBuffer *')
        self._source_buffer = None

    def __enter__(self):
        if self._entered:
            raise ValueError('cannot __enter__ multiple times')
        if self._closed:
            raise ValueError('stream is closed')
        self._entered = True
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self._entered = False
        self._decompressor = None
        self.close()
        self._source = None
        return False

    def readable(self):
        return True

    def writable(self):
        return False

    def seekable(self):
        return False

    def readline(self, size=-1):
        raise io.UnsupportedOperation()

    def readlines(self, hint=-1):
        raise io.UnsupportedOperation()

    def write(self, data):
        raise io.UnsupportedOperation()

    def writelines(self, lines):
        raise io.UnsupportedOperation()

    def isatty(self):
        return False

    def flush(self):
        return None

    def close(self):
        if self._closed:
            return None
        self._closed = True
        f = getattr(self._source, 'close', None)
        if self._closefd and f:
            f()

    @property
    def closed(self):
        return self._closed

    def tell(self):
        return self._bytes_decompressed

    def readall(self):
        chunks = []
        while True:
            chunk = self.read(1048576)
            if not chunk:
                break
            chunks.append(chunk)
        return b''.join(chunks)

    def __iter__(self):
        raise io.UnsupportedOperation()

    def __next__(self):
        raise io.UnsupportedOperation()
    next = __next__

    def _read_input(self):
        if self._in_buffer.pos < self._in_buffer.size:
            return
        if self._finished_input:
            return
        if hasattr(self._source, 'read'):
            data = self._source.read(self._read_size)
            if not data:
                self._finished_input = True
                return
            self._source_buffer = ffi.from_buffer(data)
            self._in_buffer.src = self._source_buffer
            self._in_buffer.size = len(self._source_buffer)
            self._in_buffer.pos = 0
        else:
            self._source_buffer = ffi.from_buffer(self._source)
            self._in_buffer.src = self._source_buffer
            self._in_buffer.size = len(self._source_buffer)
            self._in_buffer.pos = 0

    def _decompress_into_buffer(self, out_buffer):
        """Decompress available input into an output buffer.

        Returns True if data in output buffer should be emitted.
        """
        zresult = lib.ZSTD_decompressStream(self._decompressor._dctx, out_buffer, self._in_buffer)
        if self._in_buffer.pos == self._in_buffer.size:
            self._in_buffer.src = ffi.NULL
            self._in_buffer.pos = 0
            self._in_buffer.size = 0
            self._source_buffer = None
            if not hasattr(self._source, 'read'):
                self._finished_input = True
        if lib.ZSTD_isError(zresult):
            raise ZstdError('zstd decompress error: %s' % _zstd_error(zresult))
        return out_buffer.pos and (out_buffer.pos == out_buffer.size or (zresult == 0 and (not self._read_across_frames)))

    def read(self, size=-1):
        if self._closed:
            raise ValueError('stream is closed')
        if size < -1:
            raise ValueError('cannot read negative amounts less than -1')
        if size == -1:
            return self.readall()
        if self._finished_output or size == 0:
            return b''
        dst_buffer = ffi.new('char[]', size)
        out_buffer = ffi.new('ZSTD_outBuffer *')
        out_buffer.dst = dst_buffer
        out_buffer.size = size
        out_buffer.pos = 0
        self._read_input()
        if self._decompress_into_buffer(out_buffer):
            self._bytes_decompressed += out_buffer.pos
            return ffi.buffer(out_buffer.dst, out_buffer.pos)[:]
        while not self._finished_input:
            self._read_input()
            if self._decompress_into_buffer(out_buffer):
                self._bytes_decompressed += out_buffer.pos
                return ffi.buffer(out_buffer.dst, out_buffer.pos)[:]
        self._bytes_decompressed += out_buffer.pos
        return ffi.buffer(out_buffer.dst, out_buffer.pos)[:]

    def readinto(self, b):
        if self._closed:
            raise ValueError('stream is closed')
        if self._finished_output:
            return 0
        dest_buffer = ffi.from_buffer(b)
        ffi.memmove(b, b'', 0)
        out_buffer = ffi.new('ZSTD_outBuffer *')
        out_buffer.dst = dest_buffer
        out_buffer.size = len(dest_buffer)
        out_buffer.pos = 0
        self._read_input()
        if self._decompress_into_buffer(out_buffer):
            self._bytes_decompressed += out_buffer.pos
            return out_buffer.pos
        while not self._finished_input:
            self._read_input()
            if self._decompress_into_buffer(out_buffer):
                self._bytes_decompressed += out_buffer.pos
                return out_buffer.pos
        self._bytes_decompressed += out_buffer.pos
        return out_buffer.pos

    def read1(self, size=-1):
        if self._closed:
            raise ValueError('stream is closed')
        if size < -1:
            raise ValueError('cannot read negative amounts less than -1')
        if self._finished_output or size == 0:
            return b''
        if size == -1:
            size = DECOMPRESSION_RECOMMENDED_OUTPUT_SIZE
        dst_buffer = ffi.new('char[]', size)
        out_buffer = ffi.new('ZSTD_outBuffer *')
        out_buffer.dst = dst_buffer
        out_buffer.size = size
        out_buffer.pos = 0
        while not self._finished_input:
            self._read_input()
            self._decompress_into_buffer(out_buffer)
            if out_buffer.pos:
                break
        self._bytes_decompressed += out_buffer.pos
        return ffi.buffer(out_buffer.dst, out_buffer.pos)[:]

    def readinto1(self, b):
        if self._closed:
            raise ValueError('stream is closed')
        if self._finished_output:
            return 0
        dest_buffer = ffi.from_buffer(b)
        ffi.memmove(b, b'', 0)
        out_buffer = ffi.new('ZSTD_outBuffer *')
        out_buffer.dst = dest_buffer
        out_buffer.size = len(dest_buffer)
        out_buffer.pos = 0
        while not self._finished_input and (not self._finished_output):
            self._read_input()
            self._decompress_into_buffer(out_buffer)
            if out_buffer.pos:
                break
        self._bytes_decompressed += out_buffer.pos
        return out_buffer.pos

    def seek(self, pos, whence=os.SEEK_SET):
        if self._closed:
            raise ValueError('stream is closed')
        read_amount = 0
        if whence == os.SEEK_SET:
            if pos < 0:
                raise OSError('cannot seek to negative position with SEEK_SET')
            if pos < self._bytes_decompressed:
                raise OSError('cannot seek zstd decompression stream backwards')
            read_amount = pos - self._bytes_decompressed
        elif whence == os.SEEK_CUR:
            if pos < 0:
                raise OSError('cannot seek zstd decompression stream backwards')
            read_amount = pos
        elif whence == os.SEEK_END:
            raise OSError('zstd decompression streams cannot be seeked with SEEK_END')
        while read_amount:
            result = self.read(min(read_amount, DECOMPRESSION_RECOMMENDED_OUTPUT_SIZE))
            if not result:
                break
            read_amount -= len(result)
        return self._bytes_decompressed