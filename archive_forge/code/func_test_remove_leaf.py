import pytest
from networkx.utils.mapped_queue import MappedQueue, _HeapElement
def test_remove_leaf(self):
    h = [0, 2, 1, 6, 3, 5, 4]
    h_removed = [0, 2, 1, 6, 4, 5]
    q = self._make_mapped_queue(h)
    removed = q.remove(3)
    assert q.heap == h_removed