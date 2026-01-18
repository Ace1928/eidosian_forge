import pytest
from networkx.utils.mapped_queue import MappedQueue, _HeapElement
def test_siftdown_single(self):
    h = [1, 0]
    h_sifted = [0, 1]
    q = self._make_mapped_queue(h)
    q._siftdown(0, len(h) - 1)
    assert q.heap == h_sifted
    self._check_map(q)