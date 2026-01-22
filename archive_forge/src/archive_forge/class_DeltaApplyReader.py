from io import BytesIO
import mmap
import os
import sys
import zlib
from gitdb.fun import (
from gitdb.util import (
from gitdb.const import NULL_BYTE, BYTE_SPACE
from gitdb.utils.encoding import force_bytes
class DeltaApplyReader(LazyMixin):
    """A reader which dynamically applies pack deltas to a base object, keeping the
    memory demands to a minimum.

    The size of the final object is only obtainable once all deltas have been
    applied, unless it is retrieved from a pack index.

    The uncompressed Delta has the following layout (MSB being a most significant
    bit encoded dynamic size):

    * MSB Source Size - the size of the base against which the delta was created
    * MSB Target Size - the size of the resulting data after the delta was applied
    * A list of one byte commands (cmd) which are followed by a specific protocol:

     * cmd & 0x80 - copy delta_data[offset:offset+size]

      * Followed by an encoded offset into the delta data
      * Followed by an encoded size of the chunk to copy

     *  cmd & 0x7f - insert

      * insert cmd bytes from the delta buffer into the output stream

     * cmd == 0 - invalid operation ( or error in delta stream )
    """
    __slots__ = ('_bstream', '_dstreams', '_mm_target', '_size', '_br')
    k_max_memory_move = 250 * 1000 * 1000

    def __init__(self, stream_list):
        """Initialize this instance with a list of streams, the first stream being
        the delta to apply on top of all following deltas, the last stream being the
        base object onto which to apply the deltas"""
        assert len(stream_list) > 1, 'Need at least one delta and one base stream'
        self._bstream = stream_list[-1]
        self._dstreams = tuple(stream_list[:-1])
        self._br = 0

    def _set_cache_too_slow_without_c(self, attr):
        if len(self._dstreams) == 1:
            return self._set_cache_brute_(attr)
        dcl = connect_deltas(self._dstreams)
        if dcl.rbound() == 0:
            self._size = 0
            self._mm_target = allocate_memory(0)
            return
        self._size = dcl.rbound()
        self._mm_target = allocate_memory(self._size)
        bbuf = allocate_memory(self._bstream.size)
        stream_copy(self._bstream.read, bbuf.write, self._bstream.size, 256 * mmap.PAGESIZE)
        write = self._mm_target.write
        dcl.apply(bbuf, write)
        self._mm_target.seek(0)

    def _set_cache_brute_(self, attr):
        """If we are here, we apply the actual deltas"""
        buffer_info_list = list()
        max_target_size = 0
        for dstream in self._dstreams:
            buf = dstream.read(512)
            offset, src_size = msb_size(buf)
            offset, target_size = msb_size(buf, offset)
            buffer_info_list.append((buf[offset:], offset, src_size, target_size))
            max_target_size = max(max_target_size, target_size)
        base_size = self._bstream.size
        target_size = max_target_size
        if len(self._dstreams) > 1:
            base_size = target_size = max(base_size, max_target_size)
        bbuf = allocate_memory(base_size)
        stream_copy(self._bstream.read, bbuf.write, base_size, 256 * mmap.PAGESIZE)
        tbuf = allocate_memory(target_size)
        final_target_size = None
        for (dbuf, offset, src_size, target_size), dstream in zip(reversed(buffer_info_list), reversed(self._dstreams)):
            ddata = allocate_memory(dstream.size - offset)
            ddata.write(dbuf)
            stream_copy(dstream.read, ddata.write, dstream.size, 256 * mmap.PAGESIZE)
            if 'c_apply_delta' in globals():
                c_apply_delta(bbuf, ddata, tbuf)
            else:
                apply_delta_data(bbuf, src_size, ddata, len(ddata), tbuf.write)
            bbuf, tbuf = (tbuf, bbuf)
            bbuf.seek(0)
            tbuf.seek(0)
            final_target_size = target_size
        self._mm_target = bbuf
        self._size = final_target_size
    if not has_perf_mod:
        _set_cache_ = _set_cache_brute_
    else:
        _set_cache_ = _set_cache_too_slow_without_c

    def read(self, count=0):
        bl = self._size - self._br
        if count < 1 or count > bl:
            count = bl
        data = self._mm_target.read(count)
        self._br += len(data)
        return data

    def seek(self, offset, whence=getattr(os, 'SEEK_SET', 0)):
        """Allows to reset the stream to restart reading

        :raise ValueError: If offset and whence are not 0"""
        if offset != 0 or whence != getattr(os, 'SEEK_SET', 0):
            raise ValueError('Can only seek to position 0')
        self._br = 0
        self._mm_target.seek(0)

    @classmethod
    def new(cls, stream_list):
        """
        Convert the given list of streams into a stream which resolves deltas
        when reading from it.

        :param stream_list: two or more stream objects, first stream is a Delta
            to the object that you want to resolve, followed by N additional delta
            streams. The list's last stream must be a non-delta stream.

        :return: Non-Delta OPackStream object whose stream can be used to obtain
            the decompressed resolved data
        :raise ValueError: if the stream list cannot be handled"""
        if len(stream_list) < 2:
            raise ValueError('Need at least two streams')
        if stream_list[-1].type_id in delta_types:
            raise ValueError('Cannot resolve deltas if there is no base object stream, last one was type: %s' % stream_list[-1].type)
        return cls(stream_list)

    @property
    def type(self):
        return self._bstream.type

    @property
    def type_id(self):
        return self._bstream.type_id

    @property
    def size(self):
        """:return: number of uncompressed bytes in the stream"""
        return self._size