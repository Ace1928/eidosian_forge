import pytest
from networkx.utils.mapped_queue import MappedQueue, _HeapElement
class TestMappedDict(TestMappedQueue):

    def _make_mapped_queue(self, h):
        priority_dict = {elt: elt for elt in h}
        return MappedQueue(priority_dict)

    def test_init(self):
        d = {5: 0, 4: 1, 'a': 2, 2: 3, 1: 4}
        q = MappedQueue(d)
        assert q.position == d

    def test_ties(self):
        d = {5: 0, 4: 1, 3: 2, 2: 3, 1: 4}
        q = MappedQueue(d)
        assert q.position == {elt: pos for pos, elt in enumerate(q.heap)}

    def test_pop(self):
        d = {5: 0, 4: 1, 3: 2, 2: 3, 1: 4}
        q = MappedQueue(d)
        assert q.pop() == _HeapElement(0, 5)
        assert q.position == {elt: pos for pos, elt in enumerate(q.heap)}

    def test_empty_pop(self):
        q = MappedQueue()
        pytest.raises(IndexError, q.pop)

    def test_incomparable_ties(self):
        d = {5: 0, 4: 0, 'a': 0, 2: 0, 1: 0}
        pytest.raises(TypeError, MappedQueue, d)

    def test_push(self):
        to_push = [6, 1, 4, 3, 2, 5, 0]
        h_sifted = [0, 2, 1, 6, 3, 5, 4]
        q = MappedQueue()
        for elt in to_push:
            q.push(elt, priority=elt)
        assert q.heap == h_sifted
        self._check_map(q)

    def test_push_duplicate(self):
        to_push = [2, 1, 0]
        h_sifted = [0, 2, 1]
        q = MappedQueue()
        for elt in to_push:
            inserted = q.push(elt, priority=elt)
            assert inserted
        assert q.heap == h_sifted
        self._check_map(q)
        inserted = q.push(1, priority=1)
        assert not inserted

    def test_update_leaf(self):
        h = [0, 20, 10, 60, 30, 50, 40]
        h_updated = [0, 15, 10, 60, 20, 50, 40]
        q = self._make_mapped_queue(h)
        removed = q.update(30, 15, priority=15)
        assert q.heap == h_updated

    def test_update_root(self):
        h = [0, 20, 10, 60, 30, 50, 40]
        h_updated = [10, 20, 35, 60, 30, 50, 40]
        q = self._make_mapped_queue(h)
        removed = q.update(0, 35, priority=35)
        assert q.heap == h_updated