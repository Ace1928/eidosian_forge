from __future__ import absolute_import, unicode_literals
import io
import os
from ._cffi import (  # type: ignore
class ZstdCompressionWriter(object):
    """Writable compressing stream wrapper.

    ``ZstdCompressionWriter`` is a write-only stream interface for writing
    compressed data to another stream.

    This type conforms to the ``io.RawIOBase`` interface and should be usable
    by any type that operates against a *file-object* (``typing.BinaryIO``
    in Python type hinting speak). Only methods that involve writing will do
    useful things.

    As data is written to this stream (e.g. via ``write()``), that data
    is sent to the compressor. As compressed data becomes available from
    the compressor, it is sent to the underlying stream by calling its
    ``write()`` method.

    Both ``write()`` and ``flush()`` return the number of bytes written to the
    object's ``write()``. In many cases, small inputs do not accumulate enough
    data to cause a write and ``write()`` will return ``0``.

    Calling ``close()`` will mark the stream as closed and subsequent I/O
    operations will raise ``ValueError`` (per the documented behavior of
    ``io.RawIOBase``). ``close()`` will also call ``close()`` on the underlying
    stream if such a method exists and the instance was constructed with
    ``closefd=True``

    Instances are obtained by calling :py:meth:`ZstdCompressor.stream_writer`.

    Typically usage is as follows:

    >>> cctx = zstandard.ZstdCompressor(level=10)
    >>> compressor = cctx.stream_writer(fh)
    >>> compressor.write(b"chunk 0\\n")
    >>> compressor.write(b"chunk 1\\n")
    >>> compressor.flush()
    >>> # Receiver will be able to decode ``chunk 0\\nchunk 1\\n`` at this point.
    >>> # Receiver is also expecting more data in the zstd *frame*.
    >>>
    >>> compressor.write(b"chunk 2\\n")
    >>> compressor.flush(zstandard.FLUSH_FRAME)
    >>> # Receiver will be able to decode ``chunk 0\\nchunk 1\\nchunk 2``.
    >>> # Receiver is expecting no more data, as the zstd frame is closed.
    >>> # Any future calls to ``write()`` at this point will construct a new
    >>> # zstd frame.

    Instances can be used as context managers. Exiting the context manager is
    the equivalent of calling ``close()``, which is equivalent to calling
    ``flush(zstandard.FLUSH_FRAME)``:

    >>> cctx = zstandard.ZstdCompressor(level=10)
    >>> with cctx.stream_writer(fh) as compressor:
    ...     compressor.write(b'chunk 0')
    ...     compressor.write(b'chunk 1')
    ...     ...

    .. important::

       If ``flush(FLUSH_FRAME)`` is not called, emitted data doesn't
       constitute a full zstd *frame* and consumers of this data may complain
       about malformed input. It is recommended to use instances as a context
       manager to ensure *frames* are properly finished.

    If the size of the data being fed to this streaming compressor is known,
    you can declare it before compression begins:

    >>> cctx = zstandard.ZstdCompressor()
    >>> with cctx.stream_writer(fh, size=data_len) as compressor:
    ...     compressor.write(chunk0)
    ...     compressor.write(chunk1)
    ...     ...

    Declaring the size of the source data allows compression parameters to
    be tuned. And if ``write_content_size`` is used, it also results in the
    content size being written into the frame header of the output data.

    The size of chunks being ``write()`` to the destination can be specified:

    >>> cctx = zstandard.ZstdCompressor()
    >>> with cctx.stream_writer(fh, write_size=32768) as compressor:
    ...     ...

    To see how much memory is being used by the streaming compressor:

    >>> cctx = zstandard.ZstdCompressor()
    >>> with cctx.stream_writer(fh) as compressor:
    ...     ...
    ...     byte_size = compressor.memory_size()

    Thte total number of bytes written so far are exposed via ``tell()``:

    >>> cctx = zstandard.ZstdCompressor()
    >>> with cctx.stream_writer(fh) as compressor:
    ...     ...
    ...     total_written = compressor.tell()

    ``stream_writer()`` accepts a ``write_return_read`` boolean argument to
    control the return value of ``write()``. When ``False`` (the default),
    ``write()`` returns the number of bytes that were ``write()``'en to the
    underlying object. When ``True``, ``write()`` returns the number of bytes
    read from the input that were subsequently written to the compressor.
    ``True`` is the *proper* behavior for ``write()`` as specified by the
    ``io.RawIOBase`` interface and will become the default value in a future
    release.
    """

    def __init__(self, compressor, writer, source_size, write_size, write_return_read, closefd=True):
        self._compressor = compressor
        self._writer = writer
        self._write_size = write_size
        self._write_return_read = bool(write_return_read)
        self._closefd = bool(closefd)
        self._entered = False
        self._closing = False
        self._closed = False
        self._bytes_compressed = 0
        self._dst_buffer = ffi.new('char[]', write_size)
        self._out_buffer = ffi.new('ZSTD_outBuffer *')
        self._out_buffer.dst = self._dst_buffer
        self._out_buffer.size = len(self._dst_buffer)
        self._out_buffer.pos = 0
        zresult = lib.ZSTD_CCtx_setPledgedSrcSize(compressor._cctx, source_size)
        if lib.ZSTD_isError(zresult):
            raise ZstdError('error setting source size: %s' % _zstd_error(zresult))

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
        self._compressor = None
        return False

    def __iter__(self):
        raise io.UnsupportedOperation()

    def __next__(self):
        raise io.UnsupportedOperation()

    def memory_size(self):
        return lib.ZSTD_sizeof_CCtx(self._compressor._cctx)

    def fileno(self):
        f = getattr(self._writer, 'fileno', None)
        if f:
            return f()
        else:
            raise OSError('fileno not available on underlying writer')

    def close(self):
        if self._closed:
            return
        try:
            self._closing = True
            self.flush(FLUSH_FRAME)
        finally:
            self._closing = False
            self._closed = True
        f = getattr(self._writer, 'close', None)
        if self._closefd and f:
            f()

    @property
    def closed(self):
        return self._closed

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

    def truncate(self, size=None):
        raise io.UnsupportedOperation()

    def writable(self):
        return True

    def writelines(self, lines):
        raise NotImplementedError('writelines() is not yet implemented')

    def read(self, size=-1):
        raise io.UnsupportedOperation()

    def readall(self):
        raise io.UnsupportedOperation()

    def readinto(self, b):
        raise io.UnsupportedOperation()

    def write(self, data):
        """Send data to the compressor and possibly to the inner stream."""
        if self._closed:
            raise ValueError('stream is closed')
        total_write = 0
        data_buffer = ffi.from_buffer(data)
        in_buffer = ffi.new('ZSTD_inBuffer *')
        in_buffer.src = data_buffer
        in_buffer.size = len(data_buffer)
        in_buffer.pos = 0
        out_buffer = self._out_buffer
        out_buffer.pos = 0
        while in_buffer.pos < in_buffer.size:
            zresult = lib.ZSTD_compressStream2(self._compressor._cctx, out_buffer, in_buffer, lib.ZSTD_e_continue)
            if lib.ZSTD_isError(zresult):
                raise ZstdError('zstd compress error: %s' % _zstd_error(zresult))
            if out_buffer.pos:
                self._writer.write(ffi.buffer(out_buffer.dst, out_buffer.pos)[:])
                total_write += out_buffer.pos
                self._bytes_compressed += out_buffer.pos
                out_buffer.pos = 0
        if self._write_return_read:
            return in_buffer.pos
        else:
            return total_write

    def flush(self, flush_mode=FLUSH_BLOCK):
        """Evict data from compressor's internal state and write it to inner stream.

        Calling this method may result in 0 or more ``write()`` calls to the
        inner stream.

        This method will also call ``flush()`` on the inner stream, if such a
        method exists.

        :param flush_mode:
           How to flush the zstd compressor.

           ``zstandard.FLUSH_BLOCK`` will flush data already sent to the
           compressor but not emitted to the inner stream. The stream is still
           writable after calling this. This is the default behavior.

           See documentation for other ``zstandard.FLUSH_*`` constants for more
           flushing options.
        :return:
           Integer number of bytes written to the inner stream.
        """
        if flush_mode == FLUSH_BLOCK:
            flush = lib.ZSTD_e_flush
        elif flush_mode == FLUSH_FRAME:
            flush = lib.ZSTD_e_end
        else:
            raise ValueError('unknown flush_mode: %r' % flush_mode)
        if self._closed:
            raise ValueError('stream is closed')
        total_write = 0
        out_buffer = self._out_buffer
        out_buffer.pos = 0
        in_buffer = ffi.new('ZSTD_inBuffer *')
        in_buffer.src = ffi.NULL
        in_buffer.size = 0
        in_buffer.pos = 0
        while True:
            zresult = lib.ZSTD_compressStream2(self._compressor._cctx, out_buffer, in_buffer, flush)
            if lib.ZSTD_isError(zresult):
                raise ZstdError('zstd compress error: %s' % _zstd_error(zresult))
            if out_buffer.pos:
                self._writer.write(ffi.buffer(out_buffer.dst, out_buffer.pos)[:])
                total_write += out_buffer.pos
                self._bytes_compressed += out_buffer.pos
                out_buffer.pos = 0
            if not zresult:
                break
        f = getattr(self._writer, 'flush', None)
        if f and (not self._closing):
            f()
        return total_write

    def tell(self):
        return self._bytes_compressed