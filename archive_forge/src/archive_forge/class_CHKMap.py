import heapq
import threading
from typing import Callable
from .. import errors, lru_cache, osutils, registry, trace
from .static_tuple import StaticTuple, expect_static_tuple
class CHKMap:
    """A persistent map from string to string backed by a CHK store."""
    __slots__ = ('_store', '_root_node', '_search_key_func')

    def __init__(self, store, root_key, search_key_func=None):
        """Create a CHKMap object.

        :param store: The store the CHKMap is stored in.
        :param root_key: The root key of the map. None to create an empty
            CHKMap.
        :param search_key_func: A function mapping a key => bytes. These bytes
            are then used by the internal nodes to split up leaf nodes into
            multiple pages.
        """
        self._store = store
        if search_key_func is None:
            search_key_func = _search_key_plain
        self._search_key_func = search_key_func
        if root_key is None:
            self._root_node = LeafNode(search_key_func=search_key_func)
        else:
            self._root_node = self._node_key(root_key)

    def apply_delta(self, delta):
        """Apply a delta to the map.

        :param delta: An iterable of old_key, new_key, new_value tuples.
            If new_key is not None, then new_key->new_value is inserted
            into the map; if old_key is not None, then the old mapping
            of old_key is removed.
        """
        has_deletes = False
        as_st = StaticTuple.from_sequence
        new_items = {as_st(key) for old, key, value in delta if key is not None and old is None}
        existing_new = list(self.iteritems(key_filter=new_items))
        if existing_new:
            raise errors.InconsistentDeltaDelta(delta, 'New items are already in the map %r.' % existing_new)
        for old, new, value in delta:
            if old is not None and old != new:
                self.unmap(old, check_remap=False)
                has_deletes = True
        for old, new, value in delta:
            if new is not None:
                self.map(new, value)
        if has_deletes:
            self._check_remap()
        return self._save()

    def _ensure_root(self):
        """Ensure that the root node is an object not a key."""
        if isinstance(self._root_node, StaticTuple):
            self._root_node = self._get_node(self._root_node)

    def _get_node(self, node):
        """Get a node.

        Note that this does not update the _items dict in objects containing a
        reference to this node. As such it does not prevent subsequent IO being
        performed.

        :param node: A tuple key or node object.
        :return: A node object.
        """
        if isinstance(node, StaticTuple):
            bytes = self._read_bytes(node)
            return _deserialise(bytes, node, search_key_func=self._search_key_func)
        else:
            return node

    def _read_bytes(self, key):
        try:
            return _get_cache()[key]
        except KeyError:
            stream = self._store.get_record_stream([key], 'unordered', True)
            bytes = next(stream).get_bytes_as('fulltext')
            _get_cache()[key] = bytes
            return bytes

    def _dump_tree(self, include_keys=False, encoding='utf-8'):
        """Return the tree in a string representation."""
        self._ensure_root()

        def decode(x):
            return x.decode(encoding)
        res = self._dump_tree_node(self._root_node, prefix=b'', indent='', decode=decode, include_keys=include_keys)
        res.append('')
        return '\n'.join(res)

    def _dump_tree_node(self, node, prefix, indent, decode, include_keys=True):
        """For this node and all children, generate a string representation."""
        result = []
        if not include_keys:
            key_str = ''
        else:
            node_key = node.key()
            if node_key is not None:
                key_str = ' {}'.format(decode(node_key[0]))
            else:
                key_str = ' None'
        result.append('{}{!r} {}{}'.format(indent, decode(prefix), node.__class__.__name__, key_str))
        if isinstance(node, InternalNode):
            list(node._iter_nodes(self._store))
            for prefix, sub in sorted(node._items.items()):
                result.extend(self._dump_tree_node(sub, prefix, indent + '  ', decode=decode, include_keys=include_keys))
        else:
            for key, value in sorted(node._items.items()):
                result.append('      {!r} {!r}'.format(tuple([decode(ke) for ke in key]), decode(value)))
        return result

    @classmethod
    def from_dict(klass, store, initial_value, maximum_size=0, key_width=1, search_key_func=None):
        """Create a CHKMap in store with initial_value as the content.

        :param store: The store to record initial_value in, a VersionedFiles
            object with 1-tuple keys supporting CHK key generation.
        :param initial_value: A dict to store in store. Its keys and values
            must be bytestrings.
        :param maximum_size: The maximum_size rule to apply to nodes. This
            determines the size at which no new data is added to a single node.
        :param key_width: The number of elements in each key_tuple being stored
            in this map.
        :param search_key_func: A function mapping a key => bytes. These bytes
            are then used by the internal nodes to split up leaf nodes into
            multiple pages.
        :return: The root chk of the resulting CHKMap.
        """
        root_key = klass._create_directly(store, initial_value, maximum_size=maximum_size, key_width=key_width, search_key_func=search_key_func)
        if not isinstance(root_key, StaticTuple):
            raise AssertionError('we got a %s instead of a StaticTuple' % (type(root_key),))
        return root_key

    @classmethod
    def _create_via_map(klass, store, initial_value, maximum_size=0, key_width=1, search_key_func=None):
        result = klass(store, None, search_key_func=search_key_func)
        result._root_node.set_maximum_size(maximum_size)
        result._root_node._key_width = key_width
        delta = []
        for key, value in initial_value.items():
            delta.append((None, key, value))
        root_key = result.apply_delta(delta)
        return root_key

    @classmethod
    def _create_directly(klass, store, initial_value, maximum_size=0, key_width=1, search_key_func=None):
        node = LeafNode(search_key_func=search_key_func)
        node.set_maximum_size(maximum_size)
        node._key_width = key_width
        as_st = StaticTuple.from_sequence
        node._items = {as_st(key): val for key, val in initial_value.items()}
        node._raw_size = sum((node._key_value_len(key, value) for key, value in node._items.items()))
        node._len = len(node._items)
        node._compute_search_prefix()
        node._compute_serialised_prefix()
        if node._len > 1 and maximum_size and (node._current_size() > maximum_size):
            prefix, node_details = node._split(store)
            if len(node_details) == 1:
                raise AssertionError('Failed to split using node._split')
            node = InternalNode(prefix, search_key_func=search_key_func)
            node.set_maximum_size(maximum_size)
            node._key_width = key_width
            for split, subnode in node_details:
                node.add_node(split, subnode)
        keys = list(node.serialise(store))
        return keys[-1]

    def iter_changes(self, basis):
        """Iterate over the changes between basis and self.

        :return: An iterator of tuples: (key, old_value, new_value). Old_value
            is None for keys only in self; new_value is None for keys only in
            basis.
        """
        if self._node_key(self._root_node) == self._node_key(basis._root_node):
            return
        self._ensure_root()
        basis._ensure_root()
        excluded_keys = set()
        self_node = self._root_node
        basis_node = basis._root_node
        self_pending = []
        basis_pending = []

        def process_node(node, path, a_map, pending):
            node = a_map._get_node(node)
            if isinstance(node, LeafNode):
                path = (node._key, path)
                for key, value in node._items.items():
                    search_key = node._search_key_func(key)
                    heapq.heappush(pending, (search_key, key, value, path))
            else:
                path = (node._key, path)
                for prefix, child in node._items.items():
                    heapq.heappush(pending, (prefix, None, child, path))

        def process_common_internal_nodes(self_node, basis_node):
            self_items = set(self_node._items.items())
            basis_items = set(basis_node._items.items())
            path = (self_node._key, None)
            for prefix, child in self_items - basis_items:
                heapq.heappush(self_pending, (prefix, None, child, path))
            path = (basis_node._key, None)
            for prefix, child in basis_items - self_items:
                heapq.heappush(basis_pending, (prefix, None, child, path))

        def process_common_leaf_nodes(self_node, basis_node):
            self_items = set(self_node._items.items())
            basis_items = set(basis_node._items.items())
            path = (self_node._key, None)
            for key, value in self_items - basis_items:
                prefix = self._search_key_func(key)
                heapq.heappush(self_pending, (prefix, key, value, path))
            path = (basis_node._key, None)
            for key, value in basis_items - self_items:
                prefix = basis._search_key_func(key)
                heapq.heappush(basis_pending, (prefix, key, value, path))

        def process_common_prefix_nodes(self_node, self_path, basis_node, basis_path):
            self_node = self._get_node(self_node)
            basis_node = basis._get_node(basis_node)
            if isinstance(self_node, InternalNode) and isinstance(basis_node, InternalNode):
                process_common_internal_nodes(self_node, basis_node)
            elif isinstance(self_node, LeafNode) and isinstance(basis_node, LeafNode):
                process_common_leaf_nodes(self_node, basis_node)
            else:
                process_node(self_node, self_path, self, self_pending)
                process_node(basis_node, basis_path, basis, basis_pending)
        process_common_prefix_nodes(self_node, None, basis_node, None)
        self_seen = set()
        basis_seen = set()
        excluded_keys = set()

        def check_excluded(key_path):
            while key_path is not None:
                key, key_path = key_path
                if key in excluded_keys:
                    return True
            return False
        loop_counter = 0
        while self_pending or basis_pending:
            loop_counter += 1
            if not self_pending:
                for prefix, key, node, path in basis_pending:
                    if check_excluded(path):
                        continue
                    node = basis._get_node(node)
                    if key is not None:
                        yield (key, node, None)
                    else:
                        for key, value in node.iteritems(basis._store):
                            yield (key, value, None)
                return
            elif not basis_pending:
                for prefix, key, node, path in self_pending:
                    if check_excluded(path):
                        continue
                    node = self._get_node(node)
                    if key is not None:
                        yield (key, None, node)
                    else:
                        for key, value in node.iteritems(self._store):
                            yield (key, None, value)
                return
            elif self_pending[0][0] < basis_pending[0][0]:
                prefix, key, node, path = heapq.heappop(self_pending)
                if check_excluded(path):
                    continue
                if key is not None:
                    yield (key, None, node)
                else:
                    process_node(node, path, self, self_pending)
                    continue
            elif self_pending[0][0] > basis_pending[0][0]:
                prefix, key, node, path = heapq.heappop(basis_pending)
                if check_excluded(path):
                    continue
                if key is not None:
                    yield (key, node, None)
                else:
                    process_node(node, path, basis, basis_pending)
                    continue
            else:
                if self_pending[0][1] is None:
                    read_self = True
                else:
                    read_self = False
                if basis_pending[0][1] is None:
                    read_basis = True
                else:
                    read_basis = False
                if not read_self and (not read_basis):
                    self_details = heapq.heappop(self_pending)
                    basis_details = heapq.heappop(basis_pending)
                    if self_details[2] != basis_details[2]:
                        yield (self_details[1], basis_details[2], self_details[2])
                    continue
                if self._node_key(self_pending[0][2]) == self._node_key(basis_pending[0][2]):
                    heapq.heappop(self_pending)
                    heapq.heappop(basis_pending)
                    continue
                if read_self and read_basis:
                    self_prefix, _, self_node, self_path = heapq.heappop(self_pending)
                    basis_prefix, _, basis_node, basis_path = heapq.heappop(basis_pending)
                    if self_prefix != basis_prefix:
                        raise AssertionError('{!r} != {!r}'.format(self_prefix, basis_prefix))
                    process_common_prefix_nodes(self_node, self_path, basis_node, basis_path)
                    continue
                if read_self:
                    prefix, key, node, path = heapq.heappop(self_pending)
                    if check_excluded(path):
                        continue
                    process_node(node, path, self, self_pending)
                if read_basis:
                    prefix, key, node, path = heapq.heappop(basis_pending)
                    if check_excluded(path):
                        continue
                    process_node(node, path, basis, basis_pending)

    def iteritems(self, key_filter=None):
        """Iterate over the entire CHKMap's contents."""
        self._ensure_root()
        if key_filter is not None:
            as_st = StaticTuple.from_sequence
            key_filter = [as_st(key) for key in key_filter]
        return self._root_node.iteritems(self._store, key_filter=key_filter)

    def key(self):
        """Return the key for this map."""
        if isinstance(self._root_node, StaticTuple):
            return self._root_node
        else:
            return self._root_node._key

    def __len__(self):
        self._ensure_root()
        return len(self._root_node)

    def map(self, key, value):
        """Map a key tuple to value.

        :param key: A key to map.
        :param value: The value to assign to key.
        """
        key = StaticTuple.from_sequence(key)
        self._ensure_root()
        prefix, node_details = self._root_node.map(self._store, key, value)
        if len(node_details) == 1:
            self._root_node = node_details[0][1]
        else:
            self._root_node = InternalNode(prefix, search_key_func=self._search_key_func)
            self._root_node.set_maximum_size(node_details[0][1].maximum_size)
            self._root_node._key_width = node_details[0][1]._key_width
            for split, node in node_details:
                self._root_node.add_node(split, node)

    def _node_key(self, node):
        """Get the key for a node whether it's a tuple or node."""
        if isinstance(node, tuple):
            node = StaticTuple.from_sequence(node)
        if isinstance(node, StaticTuple):
            return node
        else:
            return node._key

    def unmap(self, key, check_remap=True):
        """remove key from the map."""
        key = StaticTuple.from_sequence(key)
        self._ensure_root()
        if isinstance(self._root_node, InternalNode):
            unmapped = self._root_node.unmap(self._store, key, check_remap=check_remap)
        else:
            unmapped = self._root_node.unmap(self._store, key)
        self._root_node = unmapped

    def _check_remap(self):
        """Check if nodes can be collapsed."""
        self._ensure_root()
        if isinstance(self._root_node, InternalNode):
            self._root_node = self._root_node._check_remap(self._store)

    def _save(self):
        """Save the map completely.

        :return: The key of the root node.
        """
        if isinstance(self._root_node, StaticTuple):
            return self._root_node
        keys = list(self._root_node.serialise(self._store))
        return keys[-1]