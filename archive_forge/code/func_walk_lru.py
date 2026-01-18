from .. import lru_cache, tests
def walk_lru(lru):
    """Test helper to walk the LRU list and assert its consistency"""
    node = lru._most_recently_used
    if node is not None:
        if node.prev is not None:
            raise AssertionError('the _most_recently_used entry is not supposed to have a previous entry %s' % (node,))
    while node is not None:
        if node.next_key is lru_cache._null_key:
            if node is not lru._least_recently_used:
                raise AssertionError('only the last node should have no next value: %s' % (node,))
            node_next = None
        else:
            node_next = lru._cache[node.next_key]
            if node_next.prev is not node:
                raise AssertionError('inconsistency found, node.next.prev != node: %s' % (node,))
        if node.prev is None:
            if node is not lru._most_recently_used:
                raise AssertionError('only the _most_recently_used should not have a previous node: %s' % (node,))
        elif node.prev.next_key != node.key:
            raise AssertionError('inconsistency found, node.prev.next != node: %s' % (node,))
        yield node
        node = node_next