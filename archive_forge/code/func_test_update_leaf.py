import pytest
from networkx.utils.mapped_queue import MappedQueue, _HeapElement
def test_update_leaf(self):
    h = [0, 20, 10, 60, 30, 50, 40]
    h_updated = [0, 15, 10, 60, 20, 50, 40]
    q = self._make_mapped_queue(h)
    removed = q.update(30, 15, priority=15)
    assert q.heap == h_updated