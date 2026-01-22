import time
import zlib
from typing import Type
from ..lazy_import import lazy_import
from breezy import (
from breezy.bzr import (
from breezy.i18n import gettext
from .. import errors, osutils, trace
from ..lru_cache import LRUSizeCache
from .btree_index import BTreeBuilder
from .versionedfile import (AbsentContentFactory, ChunkedContentFactory,
from ._groupcompress_py import (LinesDeltaIndex, apply_delta,
class _LazyGroupContentManager:
    """This manages a group of _LazyGroupCompressFactory objects."""
    _max_cut_fraction = 0.75
    _full_block_size = 4 * 1024 * 1024
    _full_mixed_block_size = 2 * 1024 * 1024
    _full_enough_block_size = 3 * 1024 * 1024
    _full_enough_mixed_block_size = 2 * 768 * 1024

    def __init__(self, block, get_compressor_settings=None):
        self._block = block
        self._factories = []
        self._last_byte = 0
        self._get_settings = get_compressor_settings
        self._compressor_settings = None

    def _get_compressor_settings(self):
        if self._compressor_settings is not None:
            return self._compressor_settings
        settings = None
        if self._get_settings is not None:
            settings = self._get_settings()
        if settings is None:
            vf = GroupCompressVersionedFiles
            settings = vf._DEFAULT_COMPRESSOR_SETTINGS
        self._compressor_settings = settings
        return self._compressor_settings

    def add_factory(self, key, parents, start, end):
        if not self._factories:
            first = True
        else:
            first = False
        factory = _LazyGroupCompressFactory(key, parents, self, start, end, first=first)
        if end > self._last_byte:
            self._last_byte = end
        self._factories.append(factory)

    def get_record_stream(self):
        """Get a record for all keys added so far."""
        for factory in self._factories:
            yield factory
            factory._bytes = None
            factory._manager = None

    def _trim_block(self, last_byte):
        """Create a new GroupCompressBlock, with just some of the content."""
        trace.mutter('stripping trailing bytes from groupcompress block %d => %d', self._block._content_length, last_byte)
        new_block = GroupCompressBlock()
        self._block._ensure_content(last_byte)
        new_block.set_content(self._block._content[:last_byte])
        self._block = new_block

    def _make_group_compressor(self):
        return GroupCompressor(self._get_compressor_settings())

    def _rebuild_block(self):
        """Create a new GroupCompressBlock with only the referenced texts."""
        compressor = self._make_group_compressor()
        tstart = time.time()
        old_length = self._block._content_length
        end_point = 0
        for factory in self._factories:
            chunks = factory.get_bytes_as('chunked')
            chunks_len = factory.size
            if chunks_len is None:
                chunks_len = sum(map(len, chunks))
            found_sha1, start_point, end_point, type = compressor.compress(factory.key, chunks, chunks_len, factory.sha1)
            factory.sha1 = found_sha1
            factory._start = start_point
            factory._end = end_point
        self._last_byte = end_point
        new_block = compressor.flush()
        delta = time.time() - tstart
        self._block = new_block
        trace.mutter('creating new compressed block on-the-fly in %.3fs %d bytes => %d bytes', delta, old_length, self._block._content_length)

    def _prepare_for_extract(self):
        """A _LazyGroupCompressFactory is about to extract to fulltext."""
        self._block._ensure_content(self._last_byte)

    def _check_rebuild_action(self):
        """Check to see if our block should be repacked."""
        total_bytes_used = 0
        last_byte_used = 0
        for factory in self._factories:
            total_bytes_used += factory._end - factory._start
            if last_byte_used < factory._end:
                last_byte_used = factory._end
        if total_bytes_used * 2 >= self._block._content_length:
            return (None, last_byte_used, total_bytes_used)
        if total_bytes_used * 2 > last_byte_used:
            return ('trim', last_byte_used, total_bytes_used)
        return ('rebuild', last_byte_used, total_bytes_used)

    def check_is_well_utilized(self):
        """Is the current block considered 'well utilized'?

        This heuristic asks if the current block considers itself to be a fully
        developed group, rather than just a loose collection of data.
        """
        if len(self._factories) == 1:
            return False
        action, last_byte_used, total_bytes_used = self._check_rebuild_action()
        block_size = self._block._content_length
        if total_bytes_used < block_size * self._max_cut_fraction:
            return False
        if block_size >= self._full_enough_block_size:
            return True
        common_prefix = None
        for factory in self._factories:
            prefix = factory.key[:-1]
            if common_prefix is None:
                common_prefix = prefix
            elif prefix != common_prefix:
                if block_size >= self._full_enough_mixed_block_size:
                    return True
                break
        return False

    def _check_rebuild_block(self):
        action, last_byte_used, total_bytes_used = self._check_rebuild_action()
        if action is None:
            return
        if action == 'trim':
            self._trim_block(last_byte_used)
        elif action == 'rebuild':
            self._rebuild_block()
        else:
            raise ValueError('unknown rebuild action: {!r}'.format(action))

    def _wire_bytes(self):
        """Return a byte stream suitable for transmitting over the wire."""
        self._check_rebuild_block()
        lines = [b'groupcompress-block\n']
        header_lines = []
        for factory in self._factories:
            key_bytes = b'\x00'.join(factory.key)
            parents = factory.parents
            if parents is None:
                parent_bytes = b'None:'
            else:
                parent_bytes = b'\t'.join((b'\x00'.join(key) for key in parents))
            record_header = b'%s\n%s\n%d\n%d\n' % (key_bytes, parent_bytes, factory._start, factory._end)
            header_lines.append(record_header)
        header_bytes = b''.join(header_lines)
        del header_lines
        header_bytes_len = len(header_bytes)
        z_header_bytes = zlib.compress(header_bytes)
        del header_bytes
        z_header_bytes_len = len(z_header_bytes)
        block_bytes_len, block_chunks = self._block.to_chunks()
        lines.append(b'%d\n%d\n%d\n' % (z_header_bytes_len, header_bytes_len, block_bytes_len))
        lines.append(z_header_bytes)
        lines.extend(block_chunks)
        del z_header_bytes, block_chunks
        return b''.join(lines)

    @classmethod
    def from_bytes(cls, bytes):
        storage_kind, z_header_len, header_len, block_len, rest = bytes.split(b'\n', 4)
        del bytes
        if storage_kind != b'groupcompress-block':
            raise ValueError('Unknown storage kind: {}'.format(storage_kind))
        z_header_len = int(z_header_len)
        if len(rest) < z_header_len:
            raise ValueError('Compressed header len shorter than all bytes')
        z_header = rest[:z_header_len]
        header_len = int(header_len)
        header = zlib.decompress(z_header)
        if len(header) != header_len:
            raise ValueError('invalid length for decompressed bytes')
        del z_header
        block_len = int(block_len)
        if len(rest) != z_header_len + block_len:
            raise ValueError('Invalid length for block')
        block_bytes = rest[z_header_len:]
        del rest
        header_lines = header.split(b'\n')
        del header
        last = header_lines.pop()
        if last != b'':
            raise ValueError('header lines did not end with a trailing newline')
        if len(header_lines) % 4 != 0:
            raise ValueError('The header was not an even multiple of 4 lines')
        block = GroupCompressBlock.from_bytes(block_bytes)
        del block_bytes
        result = cls(block)
        for start in range(0, len(header_lines), 4):
            key = tuple(header_lines[start].split(b'\x00'))
            parents_line = header_lines[start + 1]
            if parents_line == b'None:':
                parents = None
            else:
                parents = tuple([tuple(segment.split(b'\x00')) for segment in parents_line.split(b'\t') if segment])
            start_offset = int(header_lines[start + 2])
            end_offset = int(header_lines[start + 3])
            result.add_factory(key, parents, start_offset, end_offset)
        return result