import re
from bisect import bisect_right
from io import BytesIO
from ..lazy_import import lazy_import
from breezy import (
from .. import debug, errors
from .. import transport as _mod_transport
from .static_tuple import StaticTuple
class GraphIndexPrefixAdapter:
    """An adapter between GraphIndex with different key lengths.

    Queries against this will emit queries against the adapted Graph with the
    prefix added, queries for all items use iter_entries_prefix. The returned
    nodes will have their keys and node references adjusted to remove the
    prefix. Finally, an add_nodes_callback can be supplied - when called the
    nodes and references being added will have prefix prepended.
    """

    def __init__(self, adapted, prefix, missing_key_length, add_nodes_callback=None):
        """Construct an adapter against adapted with prefix."""
        self.adapted = adapted
        self.prefix_key = prefix + (None,) * missing_key_length
        self.prefix = prefix
        self.prefix_len = len(prefix)
        self.add_nodes_callback = add_nodes_callback

    def add_nodes(self, nodes):
        """Add nodes to the index.

        :param nodes: An iterable of (key, node_refs, value) entries to add.
        """
        nodes = tuple(nodes)
        translated_nodes = []
        try:
            for key, value, node_refs in nodes:
                adjusted_references = tuple((tuple((self.prefix + ref_node for ref_node in ref_list)) for ref_list in node_refs))
                translated_nodes.append((self.prefix + key, value, adjusted_references))
        except ValueError:
            for key, value in nodes:
                translated_nodes.append((self.prefix + key, value))
        self.add_nodes_callback(translated_nodes)

    def add_node(self, key, value, references=()):
        """Add a node to the index.

        :param key: The key. keys are non-empty tuples containing
            as many whitespace-free utf8 bytestrings as the key length
            defined for this index.
        :param references: An iterable of iterables of keys. Each is a
            reference to another key.
        :param value: The value to associate with the key. It may be any
            bytes as long as it does not contain \x00 or 
.
        """
        self.add_nodes(((key, value, references),))

    def _strip_prefix(self, an_iter):
        """Strip prefix data from nodes and return it."""
        for node in an_iter:
            if node[1][:self.prefix_len] != self.prefix:
                raise BadIndexData(self)
            for ref_list in node[3]:
                for ref_node in ref_list:
                    if ref_node[:self.prefix_len] != self.prefix:
                        raise BadIndexData(self)
            yield (node[0], node[1][self.prefix_len:], node[2], tuple((tuple((ref_node[self.prefix_len:] for ref_node in ref_list)) for ref_list in node[3])))

    def iter_all_entries(self):
        """Iterate over all keys within the index

        iter_all_entries is implemented against the adapted index using
        iter_entries_prefix.

        :return: An iterable of (index, key, reference_lists, value). There is no
            defined order for the result iteration - it will be in the most
            efficient order for the index (in this case dictionary hash order).
        """
        return self._strip_prefix(self.adapted.iter_entries_prefix([self.prefix_key]))

    def iter_entries(self, keys):
        """Iterate over keys within the index.

        :param keys: An iterable providing the keys to be retrieved.
        :return: An iterable of (index, key, value, reference_lists). There is no
            defined order for the result iteration - it will be in the most
            efficient order for the index (keys iteration order in this case).
        """
        return self._strip_prefix(self.adapted.iter_entries((self.prefix + key for key in keys)))

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
        return self._strip_prefix(self.adapted.iter_entries_prefix((self.prefix + key for key in keys)))

    def key_count(self):
        """Return an estimate of the number of keys in this index.

        For GraphIndexPrefixAdapter this is relatively expensive - key
        iteration with the prefix is done.
        """
        return len(list(self.iter_all_entries()))

    def validate(self):
        """Call the adapted's validate."""
        self.adapted.validate()