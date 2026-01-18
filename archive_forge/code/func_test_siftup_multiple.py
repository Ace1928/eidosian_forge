import pytest
from networkx.utils.mapped_queue import MappedQueue, _HeapElement
def test_siftup_multiple(self):
    h = [0, 1, 2, 4, 3, 5, 6]
    h_sifted = [0, 1, 2, 4, 3, 5, 6]
    q = self._make_mapped_queue(h)
    q._siftup(0)
    assert q.heap == h_sifted
    self._check_map(q)