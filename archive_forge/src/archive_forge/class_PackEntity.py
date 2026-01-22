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
class PackEntity(LazyMixin):
    """Combines the PackIndexFile and the PackFile into one, allowing the
    actual objects to be resolved and iterated"""
    __slots__ = ('_index', '_pack', '_offset_map')
    IndexFileCls = PackIndexFile
    PackFileCls = PackFile

    def __init__(self, pack_or_index_path):
        """Initialize ourselves with the path to the respective pack or index file"""
        basename, ext = os.path.splitext(pack_or_index_path)
        self._index = self.IndexFileCls('%s.idx' % basename)
        self._pack = self.PackFileCls('%s.pack' % basename)

    def close(self):
        self._index.close()
        self._pack.close()

    def _set_cache_(self, attr):
        offsets_sorted = sorted(self._index.offsets())
        last_offset = len(self._pack.data()) - self._pack.footer_size
        assert offsets_sorted, 'Cannot handle empty indices'
        offset_map = None
        if len(offsets_sorted) == 1:
            offset_map = {offsets_sorted[0]: last_offset}
        else:
            iter_offsets = iter(offsets_sorted)
            iter_offsets_plus_one = iter(offsets_sorted)
            next(iter_offsets_plus_one)
            consecutive = zip(iter_offsets, iter_offsets_plus_one)
            offset_map = dict(consecutive)
            offset_map[offsets_sorted[-1]] = last_offset
        self._offset_map = offset_map

    def _sha_to_index(self, sha):
        """:return: index for the given sha, or raise"""
        index = self._index.sha_to_index(sha)
        if index is None:
            raise BadObject(sha)
        return index

    def _iter_objects(self, as_stream):
        """Iterate over all objects in our index and yield their OInfo or OStream instences"""
        _sha = self._index.sha
        _object = self._object
        for index in range(self._index.size()):
            yield _object(_sha(index), as_stream, index)

    def _object(self, sha, as_stream, index=-1):
        """:return: OInfo or OStream object providing information about the given sha
        :param index: if not -1, its assumed to be the sha's index in the IndexFile"""
        if index < 0:
            index = self._sha_to_index(sha)
        if sha is None:
            sha = self._index.sha(index)
        offset = self._index.offset(index)
        type_id, uncomp_size, data_rela_offset = pack_object_header_info(self._pack._cursor.use_region(offset).buffer())
        if as_stream:
            if type_id not in delta_types:
                packstream = self._pack.stream(offset)
                return OStream(sha, packstream.type, packstream.size, packstream.stream)
            streams = self.collect_streams_at_offset(offset)
            dstream = DeltaApplyReader.new(streams)
            return ODeltaStream(sha, dstream.type, None, dstream)
        else:
            if type_id not in delta_types:
                return OInfo(sha, type_id_to_type_map[type_id], uncomp_size)
            streams = self.collect_streams_at_offset(offset)
            buf = streams[0].read(512)
            offset, src_size = msb_size(buf)
            offset, target_size = msb_size(buf, offset)
            if streams[-1].type_id in delta_types:
                raise BadObject(sha, 'Could not resolve delta object')
            return OInfo(sha, streams[-1].type, target_size)

    def info(self, sha):
        """Retrieve information about the object identified by the given sha

        :param sha: 20 byte sha1
        :raise BadObject:
        :return: OInfo instance, with 20 byte sha"""
        return self._object(sha, False)

    def stream(self, sha):
        """Retrieve an object stream along with its information as identified by the given sha

        :param sha: 20 byte sha1
        :raise BadObject:
        :return: OStream instance, with 20 byte sha"""
        return self._object(sha, True)

    def info_at_index(self, index):
        """As ``info``, but uses a PackIndexFile compatible index to refer to the object"""
        return self._object(None, False, index)

    def stream_at_index(self, index):
        """As ``stream``, but uses a PackIndexFile compatible index to refer to the
        object"""
        return self._object(None, True, index)

    def pack(self):
        """:return: the underlying pack file instance"""
        return self._pack

    def index(self):
        """:return: the underlying pack index file instance"""
        return self._index

    def is_valid_stream(self, sha, use_crc=False):
        """
        Verify that the stream at the given sha is valid.

        :param use_crc: if True, the index' crc is run over the compressed stream of
            the object, which is much faster than checking the sha1. It is also
            more prone to unnoticed corruption or manipulation.
        :param sha: 20 byte sha1 of the object whose stream to verify
            whether the compressed stream of the object is valid. If it is
            a delta, this only verifies that the delta's data is valid, not the
            data of the actual undeltified object, as it depends on more than
            just this stream.
            If False, the object will be decompressed and the sha generated. It must
            match the given sha

        :return: True if the stream is valid
        :raise UnsupportedOperation: If the index is version 1 only
        :raise BadObject: sha was not found"""
        if use_crc:
            if self._index.version() < 2:
                raise UnsupportedOperation("Version 1 indices do not contain crc's, verify by sha instead")
            index = self._sha_to_index(sha)
            offset = self._index.offset(index)
            next_offset = self._offset_map[offset]
            crc_value = self._index.crc(index)
            crc_update = zlib.crc32
            pack_data = self._pack.data()
            cur_pos = offset
            this_crc_value = 0
            while cur_pos < next_offset:
                rbound = min(cur_pos + chunk_size, next_offset)
                size = rbound - cur_pos
                this_crc_value = crc_update(pack_data[cur_pos:cur_pos + size], this_crc_value)
                cur_pos += size
            return this_crc_value & 4294967295 == crc_value
        else:
            shawriter = Sha1Writer()
            stream = self._object(sha, as_stream=True)
            write_object(stream.type, stream.size, stream.read, shawriter.write)
            assert shawriter.sha(as_hex=False) == sha
            return shawriter.sha(as_hex=False) == sha
        return True

    def info_iter(self):
        """
        :return: Iterator over all objects in this pack. The iterator yields
            OInfo instances"""
        return self._iter_objects(as_stream=False)

    def stream_iter(self):
        """
        :return: iterator over all objects in this pack. The iterator yields
            OStream instances"""
        return self._iter_objects(as_stream=True)

    def collect_streams_at_offset(self, offset):
        """
        As the version in the PackFile, but can resolve REF deltas within this pack
        For more info, see ``collect_streams``

        :param offset: offset into the pack file at which the object can be found"""
        streams = self._pack.collect_streams(offset)
        if streams[-1].type_id == REF_DELTA:
            stream = streams[-1]
            while stream.type_id in delta_types:
                if stream.type_id == REF_DELTA:
                    if isinstance(stream.delta_info, memoryview):
                        sindex = self._index.sha_to_index(stream.delta_info.tobytes())
                    else:
                        sindex = self._index.sha_to_index(stream.delta_info)
                    if sindex is None:
                        break
                    stream = self._pack.stream(self._index.offset(sindex))
                    streams.append(stream)
                else:
                    stream = self._pack.stream(stream.delta_info)
                    streams.append(stream)
        return streams

    def collect_streams(self, sha):
        """
        As ``PackFile.collect_streams``, but takes a sha instead of an offset.
        Additionally, ref_delta streams will be resolved within this pack.
        If this is not possible, the stream will be left alone, hence it is adivsed
        to check for unresolved ref-deltas and resolve them before attempting to
        construct a delta stream.

        :param sha: 20 byte sha1 specifying the object whose related streams you want to collect
        :return: list of streams, first being the actual object delta, the last being
            a possibly unresolved base object.
        :raise BadObject:"""
        return self.collect_streams_at_offset(self._index.offset(self._sha_to_index(sha)))

    @classmethod
    def write_pack(cls, object_iter, pack_write, index_write=None, object_count=None, zlib_compression=zlib.Z_BEST_SPEED):
        """
        Create a new pack by putting all objects obtained by the object_iterator
        into a pack which is written using the pack_write method.
        The respective index is produced as well if index_write is not Non.

        :param object_iter: iterator yielding odb output objects
        :param pack_write: function to receive strings to write into the pack stream
        :param indx_write: if not None, the function writes the index file corresponding
            to the pack.
        :param object_count: if you can provide the amount of objects in your iteration,
            this would be the place to put it. Otherwise we have to pre-iterate and store
            all items into a list to get the number, which uses more memory than necessary.
        :param zlib_compression: the zlib compression level to use
        :return: tuple(pack_sha, index_binsha) binary sha over all the contents of the pack
            and over all contents of the index. If index_write was None, index_binsha will be None

        **Note:** The destination of the write functions is up to the user. It could
        be a socket, or a file for instance

        **Note:** writes only undeltified objects"""
        objs = object_iter
        if not object_count:
            if not isinstance(object_iter, (tuple, list)):
                objs = list(object_iter)
            object_count = len(objs)
        pack_writer = FlexibleSha1Writer(pack_write)
        pwrite = pack_writer.write
        ofs = 0
        index = None
        wants_index = index_write is not None
        pwrite(pack('>LLL', PackFile.pack_signature, PackFile.pack_version_default, object_count))
        ofs += 12
        if wants_index:
            index = IndexWriter()
        actual_count = 0
        for obj in objs:
            actual_count += 1
            crc = 0
            hdr = create_pack_object_header(obj.type_id, obj.size)
            if index_write:
                crc = crc32(hdr)
            else:
                crc = None
            pwrite(hdr)
            zstream = zlib.compressobj(zlib_compression)
            ostream = obj.stream
            br, bw, crc = write_stream_to_pack(ostream.read, pwrite, zstream, base_crc=crc)
            assert br == obj.size
            if wants_index:
                index.append(obj.binsha, crc, ofs)
            ofs += len(hdr) + bw
            if actual_count == object_count:
                break
        if actual_count != object_count:
            raise ValueError('Expected to write %i objects into pack, but received only %i from iterators' % (object_count, actual_count))
        pack_sha = pack_writer.sha(as_hex=False)
        assert len(pack_sha) == 20
        pack_write(pack_sha)
        ofs += len(pack_sha)
        index_sha = None
        if wants_index:
            index_sha = index.write(pack_sha, index_write)
        return (pack_sha, index_sha)

    @classmethod
    def create(cls, object_iter, base_dir, object_count=None, zlib_compression=zlib.Z_BEST_SPEED):
        """Create a new on-disk entity comprised of a properly named pack file and a properly named
        and corresponding index file. The pack contains all OStream objects contained in object iter.
        :param base_dir: directory which is to contain the files
        :return: PackEntity instance initialized with the new pack

        **Note:** for more information on the other parameters see the write_pack method"""
        pack_fd, pack_path = tempfile.mkstemp('', 'pack', base_dir)
        index_fd, index_path = tempfile.mkstemp('', 'index', base_dir)
        pack_write = lambda d: os.write(pack_fd, d)
        index_write = lambda d: os.write(index_fd, d)
        pack_binsha, index_binsha = cls.write_pack(object_iter, pack_write, index_write, object_count, zlib_compression)
        os.close(pack_fd)
        os.close(index_fd)
        fmt = 'pack-%s.%s'
        new_pack_path = os.path.join(base_dir, fmt % (bin_to_hex(pack_binsha), 'pack'))
        new_index_path = os.path.join(base_dir, fmt % (bin_to_hex(pack_binsha), 'idx'))
        os.rename(pack_path, new_pack_path)
        os.rename(index_path, new_index_path)
        return cls(new_pack_path)