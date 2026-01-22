from __future__ import absolute_import, unicode_literals
import io
import os
from ._cffi import (  # type: ignore
class ZstdDecompressionObj(object):
    """A standard library API compatible decompressor.

    This type implements a compressor that conforms to the API by other
    decompressors in Python's standard library. e.g. ``zlib.decompressobj``
    or ``bz2.BZ2Decompressor``. This allows callers to use zstd compression
    while conforming to a similar API.

    Compressed data chunks are fed into ``decompress(data)`` and
    uncompressed output (or an empty bytes) is returned. Output from
    subsequent calls needs to be concatenated to reassemble the full
    decompressed byte sequence.

    If ``read_across_frames=False``, each instance is single use: once an
    input frame is decoded, ``decompress()`` will raise an exception. If
    ``read_across_frames=True``, instances can decode multiple frames.

    >>> dctx = zstandard.ZstdDecompressor()
    >>> dobj = dctx.decompressobj()
    >>> data = dobj.decompress(compressed_chunk_0)
    >>> data = dobj.decompress(compressed_chunk_1)

    By default, calls to ``decompress()`` write output data in chunks of size
    ``DECOMPRESSION_RECOMMENDED_OUTPUT_SIZE``. These chunks are concatenated
    before being returned to the caller. It is possible to define the size of
    these temporary chunks by passing ``write_size`` to ``decompressobj()``:

    >>> dctx = zstandard.ZstdDecompressor()
    >>> dobj = dctx.decompressobj(write_size=1048576)

    .. note::

       Because calls to ``decompress()`` may need to perform multiple
       memory (re)allocations, this streaming decompression API isn't as
       efficient as other APIs.
    """

    def __init__(self, decompressor, write_size, read_across_frames):
        self._decompressor = decompressor
        self._write_size = write_size
        self._finished = False
        self._read_across_frames = read_across_frames
        self._unused_input = b''

    def decompress(self, data):
        """Send compressed data to the decompressor and obtain decompressed data.

        :param data:
           Data to feed into the decompressor.
        :return:
           Decompressed bytes.
        """
        if self._finished:
            raise ZstdError('cannot use a decompressobj multiple times')
        in_buffer = ffi.new('ZSTD_inBuffer *')
        out_buffer = ffi.new('ZSTD_outBuffer *')
        data_buffer = ffi.from_buffer(data)
        if len(data_buffer) == 0:
            return b''
        in_buffer.src = data_buffer
        in_buffer.size = len(data_buffer)
        in_buffer.pos = 0
        dst_buffer = ffi.new('char[]', self._write_size)
        out_buffer.dst = dst_buffer
        out_buffer.size = len(dst_buffer)
        out_buffer.pos = 0
        chunks = []
        while True:
            zresult = lib.ZSTD_decompressStream(self._decompressor._dctx, out_buffer, in_buffer)
            if lib.ZSTD_isError(zresult):
                raise ZstdError('zstd decompressor error: %s' % _zstd_error(zresult))
            if out_buffer.pos:
                chunks.append(ffi.buffer(out_buffer.dst, out_buffer.pos)[:])
            if zresult == 0 and (not self._read_across_frames):
                self._finished = True
                self._decompressor = None
                self._unused_input = data[in_buffer.pos:in_buffer.size]
                break
            elif zresult == 0 and self._read_across_frames:
                if in_buffer.pos == in_buffer.size:
                    break
                else:
                    out_buffer.pos = 0
            elif in_buffer.pos == in_buffer.size and out_buffer.pos < out_buffer.size:
                break
            else:
                out_buffer.pos = 0
        return b''.join(chunks)

    def flush(self, length=0):
        """Effectively a no-op.

        Implemented for compatibility with the standard library APIs.

        Safe to call at any time.

        :return:
           Empty bytes.
        """
        return b''

    @property
    def unused_data(self):
        """Bytes past the end of compressed data.

        If ``decompress()`` is fed additional data beyond the end of a zstd
        frame, this value will be non-empty once ``decompress()`` fully decodes
        the input frame.
        """
        return self._unused_input

    @property
    def unconsumed_tail(self):
        """Data that has not yet been fed into the decompressor."""
        return b''

    @property
    def eof(self):
        """Whether the end of the compressed data stream has been reached."""
        return self._finished