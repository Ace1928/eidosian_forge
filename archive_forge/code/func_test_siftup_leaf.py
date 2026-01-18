import pytest
from networkx.utils.mapped_queue import MappedQueue, _HeapElement
def test_siftup_leaf(self):
    h = [2]
    h_sifted = [2]
    q = self._make_mapped_queue(h)
    q._siftup(0)
    assert q.heap == h_sifted
    self._check_map(q)