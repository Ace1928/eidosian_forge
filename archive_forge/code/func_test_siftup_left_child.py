import pytest
from networkx.utils.mapped_queue import MappedQueue, _HeapElement
def test_siftup_left_child(self):
    h = [2, 0, 1]
    h_sifted = [0, 2, 1]
    q = self._make_mapped_queue(h)
    q._siftup(0)
    assert q.heap == h_sifted
    self._check_map(q)