import pytest
from networkx.utils.mapped_queue import MappedQueue, _HeapElement
def test_siftup_right_child(self):
    h = [2, 1, 0]
    h_sifted = [0, 1, 2]
    q = self._make_mapped_queue(h)
    q._siftup(0)
    assert q.heap == h_sifted
    self._check_map(q)