import zlib
from gitdb.exc import (
from gitdb.util import (
from gitdb.fun import (
from gitdb.base import (      # Amazing !
from gitdb.stream import (
from struct import pack
from binascii import crc32
from gitdb.const import NULL_BYTE
import tempfile
import array
import os
import sys
class PackFile(LazyMixin):
    """A pack is a file written according to the Version 2 for git packs

    As we currently use memory maps, it could be assumed that the maximum size of
    packs therefore is 32 bit on 32 bit systems. On 64 bit systems, this should be
    fine though.

    **Note:** at some point, this might be implemented using streams as well, or
    streams are an alternate path in the case memory maps cannot be created
    for some reason - one clearly doesn't want to read 10GB at once in that
    case"""
    __slots__ = ('_packpath', '_cursor', '_size', '_version')
    pack_signature = 1346454347
    pack_version_default = 2
    first_object_offset = 3 * 4
    footer_size = 20

    def __init__(self, packpath):
        self._packpath = packpath

    def close(self):
        mman.force_map_handle_removal_win(self._packpath)
        self._cursor = None

    def _set_cache_(self, attr):
        self._cursor = mman.make_cursor(self._packpath).use_region()
        type_id, self._version, self._size = unpack_from('>LLL', self._cursor.map(), 0)
        if type_id != self.pack_signature:
            raise ParseError('Invalid pack signature: %i' % type_id)

    def _iter_objects(self, start_offset, as_stream=True):
        """Handle the actual iteration of objects within this pack"""
        c = self._cursor
        content_size = c.file_size() - self.footer_size
        cur_offset = start_offset or self.first_object_offset
        null = NullStream()
        while cur_offset < content_size:
            data_offset, ostream = pack_object_at(c, cur_offset, True)
            stream_copy(ostream.read, null.write, ostream.size, chunk_size)
            assert ostream.stream._br == ostream.size
            cur_offset += data_offset - ostream.pack_offset + ostream.stream.compressed_bytes_read()
            if as_stream:
                ostream.stream.seek(0)
            yield ostream

    def size(self):
        """:return: The amount of objects stored in this pack"""
        return self._size

    def version(self):
        """:return: the version of this pack"""
        return self._version

    def data(self):
        """
        :return: read-only data of this pack. It provides random access and usually
            is a memory map.
        :note: This method is unsafe as it returns a window into a file which might be larger than than the actual window size"""
        return self._cursor.use_region().map()

    def checksum(self):
        """:return: 20 byte sha1 hash on all object sha's contained in this file"""
        return self._cursor.use_region(self._cursor.file_size() - 20).buffer()[:]

    def path(self):
        """:return: path to the packfile"""
        return self._packpath

    def collect_streams(self, offset):
        """
        :return: list of pack streams which are required to build the object
            at the given offset. The first entry of the list is the object at offset,
            the last one is either a full object, or a REF_Delta stream. The latter
            type needs its reference object to be locked up in an ODB to form a valid
            delta chain.
            If the object at offset is no delta, the size of the list is 1.
        :param offset: specifies the first byte of the object within this pack"""
        out = list()
        c = self._cursor
        while True:
            ostream = pack_object_at(c, offset, True)[1]
            out.append(ostream)
            if ostream.type_id == OFS_DELTA:
                offset = ostream.pack_offset - ostream.delta_info
            else:
                break
        return out

    def info(self, offset):
        """Retrieve information about the object at the given file-absolute offset

        :param offset: byte offset
        :return: OPackInfo instance, the actual type differs depending on the type_id attribute"""
        return pack_object_at(self._cursor, offset or self.first_object_offset, False)[1]

    def stream(self, offset):
        """Retrieve an object at the given file-relative offset as stream along with its information

        :param offset: byte offset
        :return: OPackStream instance, the actual type differs depending on the type_id attribute"""
        return pack_object_at(self._cursor, offset or self.first_object_offset, True)[1]

    def stream_iter(self, start_offset=0):
        """
        :return: iterator yielding OPackStream compatible instances, allowing
            to access the data in the pack directly.
        :param start_offset: offset to the first object to iterate. If 0, iteration
            starts at the very first object in the pack.

        **Note:** Iterating a pack directly is costly as the datastream has to be decompressed
        to determine the bounds between the objects"""
        return self._iter_objects(start_offset, as_stream=True)