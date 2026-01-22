import pytest
import networkx as nx
from networkx.utils import BinaryHeap, PairingHeap
def _test_heap_class(cls, *args, **kwargs):
    heap = cls(*args, **kwargs)
    for op in data:
        if op[-1] is not nx.NetworkXError:
            assert op[-1] == getattr(heap, op[0])(*op[1:-1])
        else:
            pytest.raises(op[-1], getattr(heap, op[0]), *op[1:-1])
    for i in range(99, -1, -1):
        assert heap.insert(i, i)
    for i in range(50):
        assert heap.pop() == (i, i)
    for i in range(100):
        assert heap.insert(i, i) == (i < 50)
    for i in range(100):
        assert not heap.insert(i, i + 1)
    for i in range(50):
        assert heap.pop() == (i, i)
    for i in range(100):
        assert heap.insert(i, i + 1) == (i < 50)
    for i in range(49):
        assert heap.pop() == (i, i + 1)
    assert sorted([heap.pop(), heap.pop()]) == [(49, 50), (50, 50)]
    for i in range(51, 100):
        assert not heap.insert(i, i + 1, True)
    for i in range(51, 70):
        assert heap.pop() == (i, i + 1)
    for i in range(100):
        assert heap.insert(i, i)
    for i in range(100):
        assert heap.pop() == (i, i)
    pytest.raises(nx.NetworkXError, heap.pop)