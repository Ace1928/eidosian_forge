import pytest
from networkx.utils.mapped_queue import MappedQueue, _HeapElement
def test_siftdown_multiple(self):
    h = [1, 2, 3, 4, 5, 6, 7, 0]
    h_sifted = [0, 1, 3, 2, 5, 6, 7, 4]
    q = self._make_mapped_queue(h)
    q._siftdown(0, len(h) - 1)
    assert q.heap == h_sifted
    self._check_map(q)