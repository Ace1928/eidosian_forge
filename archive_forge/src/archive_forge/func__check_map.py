import pytest
from networkx.utils.mapped_queue import MappedQueue, _HeapElement
def _check_map(self, q):
    assert q.position == {elt: pos for pos, elt in enumerate(q.heap)}