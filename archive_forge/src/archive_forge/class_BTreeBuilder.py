from io import BytesIO
from ..lazy_import import lazy_import
import bisect
import math
import tempfile
import zlib
from .. import (chunk_writer, debug, fifo_cache, lru_cache, osutils, trace,
from . import index, static_tuple
from .index import _OPTION_KEY_ELEMENTS, _OPTION_LEN, _OPTION_NODE_REFS
class BTreeBuilder(index.GraphIndexBuilder):
    """A Builder for B+Tree based Graph indices.

    The resulting graph has the structure:

    _SIGNATURE OPTIONS NODES
    _SIGNATURE     := 'B+Tree Graph Index 1' NEWLINE
    OPTIONS        := REF_LISTS KEY_ELEMENTS LENGTH
    REF_LISTS      := 'node_ref_lists=' DIGITS NEWLINE
    KEY_ELEMENTS   := 'key_elements=' DIGITS NEWLINE
    LENGTH         := 'len=' DIGITS NEWLINE
    ROW_LENGTHS    := 'row_lengths' DIGITS (COMMA DIGITS)*
    NODES          := NODE_COMPRESSED*
    NODE_COMPRESSED:= COMPRESSED_BYTES{4096}
    NODE_RAW       := INTERNAL | LEAF
    INTERNAL       := INTERNAL_FLAG POINTERS
    LEAF           := LEAF_FLAG ROWS
    KEY_ELEMENT    := Not-whitespace-utf8
    KEY            := KEY_ELEMENT (NULL KEY_ELEMENT)*
    ROWS           := ROW*
    ROW            := KEY NULL ABSENT? NULL REFERENCES NULL VALUE NEWLINE
    ABSENT         := 'a'
    REFERENCES     := REFERENCE_LIST (TAB REFERENCE_LIST){node_ref_lists - 1}
    REFERENCE_LIST := (REFERENCE (CR REFERENCE)*)?
    REFERENCE      := KEY
    VALUE          := no-newline-no-null-bytes
    """

    def __init__(self, reference_lists=0, key_elements=1, spill_at=100000):
        """See GraphIndexBuilder.__init__.

        :param spill_at: Optional parameter controlling the maximum number
            of nodes that BTreeBuilder will hold in memory.
        """
        index.GraphIndexBuilder.__init__(self, reference_lists=reference_lists, key_elements=key_elements)
        self._spill_at = spill_at
        self._backing_indices = []
        self._nodes = {}
        self._nodes_by_key = None
        self._optimize_for_size = False

    def add_node(self, key, value, references=()):
        """Add a node to the index.

        If adding the node causes the builder to reach its spill_at threshold,
        disk spilling will be triggered.

        :param key: The key. keys are non-empty tuples containing
            as many whitespace-free utf8 bytestrings as the key length
            defined for this index.
        :param references: An iterable of iterables of keys. Each is a
            reference to another key.
        :param value: The value to associate with the key. It may be any
            bytes as long as it does not contain \\0 or \\n.
        """
        key = static_tuple.StaticTuple.from_sequence(key).intern()
        node_refs, _ = self._check_key_ref_value(key, references, value)
        if key in self._nodes:
            raise index.BadIndexDuplicateKey(key, self)
        self._nodes[key] = static_tuple.StaticTuple(node_refs, value)
        if self._nodes_by_key is not None and self._key_length > 1:
            self._update_nodes_by_key(key, value, node_refs)
        if len(self._nodes) < self._spill_at:
            return
        self._spill_mem_keys_to_disk()

    def _spill_mem_keys_to_disk(self):
        """Write the in memory keys down to disk to cap memory consumption.

        If we already have some keys written to disk, we will combine them so
        as to preserve the sorted order.  The algorithm for combining uses
        powers of two.  So on the first spill, write all mem nodes into a
        single index. On the second spill, combine the mem nodes with the nodes
        on disk to create a 2x sized disk index and get rid of the first index.
        On the third spill, create a single new disk index, which will contain
        the mem nodes, and preserve the existing 2x sized index.  On the fourth,
        combine mem with the first and second indexes, creating a new one of
        size 4x. On the fifth create a single new one, etc.
        """
        if self._combine_backing_indices:
            new_backing_file, size, backing_pos = self._spill_mem_keys_and_combine()
        else:
            new_backing_file, size = self._spill_mem_keys_without_combining()
        new_backing = BTreeGraphIndex(transport.get_transport_from_path('.'), '<temp>', size)
        new_backing._file = new_backing_file
        if self._combine_backing_indices:
            if len(self._backing_indices) == backing_pos:
                self._backing_indices.append(None)
            self._backing_indices[backing_pos] = new_backing
            for backing_pos in range(backing_pos):
                self._backing_indices[backing_pos] = None
        else:
            self._backing_indices.append(new_backing)
        self._nodes = {}
        self._nodes_by_key = None

    def _spill_mem_keys_without_combining(self):
        return self._write_nodes(self._iter_mem_nodes(), allow_optimize=False)

    def _spill_mem_keys_and_combine(self):
        iterators_to_combine = [self._iter_mem_nodes()]
        pos = -1
        for pos, backing in enumerate(self._backing_indices):
            if backing is None:
                pos -= 1
                break
            iterators_to_combine.append(backing.iter_all_entries())
        backing_pos = pos + 1
        new_backing_file, size = self._write_nodes(self._iter_smallest(iterators_to_combine), allow_optimize=False)
        return (new_backing_file, size, backing_pos)

    def add_nodes(self, nodes):
        """Add nodes to the index.

        :param nodes: An iterable of (key, node_refs, value) entries to add.
        """
        if self.reference_lists:
            for key, value, node_refs in nodes:
                self.add_node(key, value, node_refs)
        else:
            for key, value in nodes:
                self.add_node(key, value)

    def _iter_mem_nodes(self):
        """Iterate over the nodes held in memory."""
        nodes = self._nodes
        if self.reference_lists:
            for key in sorted(nodes):
                references, value = nodes[key]
                yield (self, key, value, references)
        else:
            for key in sorted(nodes):
                references, value = nodes[key]
                yield (self, key, value)

    def _iter_smallest(self, iterators_to_combine):
        if len(iterators_to_combine) == 1:
            yield from iterators_to_combine[0]
            return
        current_values = []
        for iterator in iterators_to_combine:
            try:
                current_values.append(next(iterator))
            except StopIteration:
                current_values.append(None)
        last = None
        while True:
            candidates = [(item[1][1], item) for item in enumerate(current_values) if item[1] is not None]
            if not len(candidates):
                return
            selected = min(candidates)
            selected = selected[1]
            if last == selected[1][1]:
                raise index.BadIndexDuplicateKey(last, self)
            last = selected[1][1]
            yield ((self,) + selected[1][1:])
            pos = selected[0]
            try:
                current_values[pos] = next(iterators_to_combine[pos])
            except StopIteration:
                current_values[pos] = None

    def _add_key(self, string_key, line, rows, allow_optimize=True):
        """Add a key to the current chunk.

        :param string_key: The key to add.
        :param line: The fully serialised key and value.
        :param allow_optimize: If set to False, prevent setting the optimize
            flag when writing out. This is used by the _spill_mem_keys_to_disk
            functionality.
        """
        new_leaf = False
        if rows[-1].writer is None:
            new_leaf = True
            for pos, internal_row in enumerate(rows[:-1]):
                if internal_row.writer is None:
                    length = _PAGE_SIZE
                    if internal_row.nodes == 0:
                        length -= _RESERVED_HEADER_BYTES
                    if allow_optimize:
                        optimize_for_size = self._optimize_for_size
                    else:
                        optimize_for_size = False
                    internal_row.writer = chunk_writer.ChunkWriter(length, 0, optimize_for_size=optimize_for_size)
                    internal_row.writer.write(_INTERNAL_FLAG)
                    internal_row.writer.write(_INTERNAL_OFFSET + b'%d\n' % rows[pos + 1].nodes)
            length = _PAGE_SIZE
            if rows[-1].nodes == 0:
                length -= _RESERVED_HEADER_BYTES
            rows[-1].writer = chunk_writer.ChunkWriter(length, optimize_for_size=self._optimize_for_size)
            rows[-1].writer.write(_LEAF_FLAG)
        if rows[-1].writer.write(line):
            if new_leaf:
                raise index.BadIndexKey(string_key)
            rows[-1].finish_node()
            key_line = string_key + b'\n'
            new_row = True
            for row in reversed(rows[:-1]):
                if row.writer.write(key_line):
                    row.finish_node()
                else:
                    new_row = False
                    break
            if new_row:
                if 'index' in debug.debug_flags:
                    trace.mutter('Inserting new global row.')
                new_row = _InternalBuilderRow()
                reserved_bytes = 0
                rows.insert(0, new_row)
                new_row.writer = chunk_writer.ChunkWriter(_PAGE_SIZE - _RESERVED_HEADER_BYTES, reserved_bytes, optimize_for_size=self._optimize_for_size)
                new_row.writer.write(_INTERNAL_FLAG)
                new_row.writer.write(_INTERNAL_OFFSET + b'%d\n' % (rows[1].nodes - 1))
                new_row.writer.write(key_line)
            self._add_key(string_key, line, rows, allow_optimize=allow_optimize)

    def _write_nodes(self, node_iterator, allow_optimize=True):
        """Write node_iterator out as a B+Tree.

        :param node_iterator: An iterator of sorted nodes. Each node should
            match the output given by iter_all_entries.
        :param allow_optimize: If set to False, prevent setting the optimize
            flag when writing out. This is used by the _spill_mem_keys_to_disk
            functionality.
        :return: A file handle for a temporary file containing a B+Tree for
            the nodes.
        """
        rows = []
        key_count = 0
        self.row_lengths = []
        for node in node_iterator:
            if key_count == 0:
                rows.append(_LeafBuilderRow())
            key_count += 1
            string_key, line = _btree_serializer._flatten_node(node, self.reference_lists)
            self._add_key(string_key, line, rows, allow_optimize=allow_optimize)
        for row in reversed(rows):
            pad = not isinstance(row, _LeafBuilderRow)
            row.finish_node(pad=pad)
        lines = [_BTSIGNATURE]
        lines.append(b'%s%d\n' % (_OPTION_NODE_REFS, self.reference_lists))
        lines.append(b'%s%d\n' % (_OPTION_KEY_ELEMENTS, self._key_length))
        lines.append(b'%s%d\n' % (_OPTION_LEN, key_count))
        row_lengths = [row.nodes for row in rows]
        lines.append(_OPTION_ROW_LENGTHS + ','.join(map(str, row_lengths)).encode('ascii') + b'\n')
        if row_lengths and row_lengths[-1] > 1:
            result = tempfile.NamedTemporaryFile(prefix='bzr-index-')
        else:
            result = BytesIO()
        result.writelines(lines)
        position = sum(map(len, lines))
        if position > _RESERVED_HEADER_BYTES:
            raise AssertionError('Could not fit the header in the reserved space: %d > %d' % (position, _RESERVED_HEADER_BYTES))
        for row in rows:
            reserved = _RESERVED_HEADER_BYTES
            row.spool.flush()
            row.spool.seek(0)
            node = row.spool.read(_PAGE_SIZE)
            result.write(node[reserved:])
            if len(node) == _PAGE_SIZE:
                result.write(b'\x00' * (reserved - position))
            position = 0
            copied_len = osutils.pumpfile(row.spool, result)
            if copied_len != (row.nodes - 1) * _PAGE_SIZE:
                if not isinstance(row, _LeafBuilderRow):
                    raise AssertionError('Incorrect amount of data copied expected: %d, got: %d' % ((row.nodes - 1) * _PAGE_SIZE, copied_len))
        result.flush()
        size = result.tell()
        result.seek(0)
        return (result, size)

    def finish(self):
        """Finalise the index.

        :return: A file handle for a temporary file containing the nodes added
            to the index.
        """
        return self._write_nodes(self.iter_all_entries())[0]

    def iter_all_entries(self):
        """Iterate over all keys within the index

        :return: An iterable of (index, key, value, reference_lists). There is
            no defined order for the result iteration - it will be in the most
            efficient order for the index (in this case dictionary hash order).
        """
        if 'evil' in debug.debug_flags:
            trace.mutter_callsite(3, 'iter_all_entries scales with size of history.')
        iterators = [self._iter_mem_nodes()]
        for backing in self._backing_indices:
            if backing is not None:
                iterators.append(backing.iter_all_entries())
        if len(iterators) == 1:
            return iterators[0]
        return self._iter_smallest(iterators)

    def iter_entries(self, keys):
        """Iterate over keys within the index.

        :param keys: An iterable providing the keys to be retrieved.
        :return: An iterable of (index, key, value, reference_lists). There is
            no defined order for the result iteration - it will be in the most
            efficient order for the index (keys iteration order in this case).
        """
        keys = set(keys)
        nodes = self._nodes
        local_keys = [key for key in keys if key in nodes]
        if self.reference_lists:
            for key in local_keys:
                node = nodes[key]
                yield (self, key, node[1], node[0])
        else:
            for key in local_keys:
                node = nodes[key]
                yield (self, key, node[1])
        if not self._backing_indices:
            return
        keys.difference_update(local_keys)
        for backing in self._backing_indices:
            if backing is None:
                continue
            if not keys:
                return
            for node in backing.iter_entries(keys):
                keys.remove(node[1])
                yield ((self,) + node[1:])

    def iter_entries_prefix(self, keys):
        """Iterate over keys within the index using prefix matching.

        Prefix matching is applied within the tuple of a key, not to within
        the bytestring of each key element. e.g. if you have the keys ('foo',
        'bar'), ('foobar', 'gam') and do a prefix search for ('foo', None) then
        only the former key is returned.

        :param keys: An iterable providing the key prefixes to be retrieved.
            Each key prefix takes the form of a tuple the length of a key, but
            with the last N elements 'None' rather than a regular bytestring.
            The first element cannot be 'None'.
        :return: An iterable as per iter_all_entries, but restricted to the
            keys with a matching prefix to those supplied. No additional keys
            will be returned, and every match that is in the index will be
            returned.
        """
        keys = set(keys)
        if not keys:
            return
        for backing in self._backing_indices:
            if backing is None:
                continue
            for node in backing.iter_entries_prefix(keys):
                yield ((self,) + node[1:])
        if self._key_length == 1:
            for key in keys:
                index._sanity_check_key(self, key)
                try:
                    node = self._nodes[key]
                except KeyError:
                    continue
                if self.reference_lists:
                    yield (self, key, node[1], node[0])
                else:
                    yield (self, key, node[1])
            return
        nodes_by_key = self._get_nodes_by_key()
        yield from index._iter_entries_prefix(self, nodes_by_key, keys)

    def _get_nodes_by_key(self):
        if self._nodes_by_key is None:
            nodes_by_key = {}
            if self.reference_lists:
                for key, (references, value) in self._nodes.items():
                    key_dict = nodes_by_key
                    for subkey in key[:-1]:
                        key_dict = key_dict.setdefault(subkey, {})
                    key_dict[key[-1]] = (key, value, references)
            else:
                for key, (references, value) in self._nodes.items():
                    key_dict = nodes_by_key
                    for subkey in key[:-1]:
                        key_dict = key_dict.setdefault(subkey, {})
                    key_dict[key[-1]] = (key, value)
            self._nodes_by_key = nodes_by_key
        return self._nodes_by_key

    def key_count(self):
        """Return an estimate of the number of keys in this index.

        For InMemoryGraphIndex the estimate is exact.
        """
        return len(self._nodes) + sum((backing.key_count() for backing in self._backing_indices if backing is not None))

    def validate(self):
        """In memory index's have no known corruption at the moment."""

    def __lt__(self, other):
        if isinstance(other, type(self)):
            return self._nodes < other._nodes
        if isinstance(other, BTreeGraphIndex):
            return False
        raise TypeError