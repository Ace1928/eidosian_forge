import struct
import zlib
from .static_tuple import StaticTuple
def _deserialise_leaf_node(data, key, search_key_func=None):
    """Deserialise bytes, with key key, into a LeafNode.

    :param bytes: The bytes of the node.
    :param key: The key that the serialised node has.
    """
    global _unknown, _LeafNode, _InternalNode
    if _LeafNode is None:
        from . import chk_map
        _unknown = chk_map._unknown
        _LeafNode = chk_map.LeafNode
        _InternalNode = chk_map.InternalNode
    result = _LeafNode(search_key_func=search_key_func)
    lines = data.split(b'\n')
    trailing = lines.pop()
    if trailing != b'':
        raise AssertionError('We did not have a final newline for %s' % (key,))
    items = {}
    if lines[0] != b'chkleaf:':
        raise ValueError('not a serialised leaf node: %r' % bytes)
    maximum_size = int(lines[1])
    width = int(lines[2])
    length = int(lines[3])
    prefix = lines[4]
    pos = 5
    while pos < len(lines):
        line = prefix + lines[pos]
        elements = line.split(b'\x00')
        pos += 1
        if len(elements) != width + 1:
            raise AssertionError('Incorrect number of elements (%d vs %d) for: %r' % (len(elements), width + 1, line))
        num_value_lines = int(elements[-1])
        value_lines = lines[pos:pos + num_value_lines]
        pos += num_value_lines
        value = b'\n'.join(value_lines)
        items[StaticTuple.from_sequence(elements[:-1])] = value
    if len(items) != length:
        raise AssertionError('item count (%d) mismatch for key %s, bytes %r' % (length, key, bytes))
    result._items = items
    result._len = length
    result._maximum_size = maximum_size
    result._key = key
    result._key_width = width
    result._raw_size = sum(map(len, lines[5:])) + length * len(prefix) + (len(lines) - 5)
    if not items:
        result._search_prefix = None
        result._common_serialised_prefix = None
    else:
        result._search_prefix = _unknown
        result._common_serialised_prefix = prefix
    if len(data) != result._current_size():
        raise AssertionError('_current_size computed incorrectly')
    return result