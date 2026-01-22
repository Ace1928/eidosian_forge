from dulwich.tests import TestCase
from ..graph import WorkList, _find_lcas, can_fast_forward
from ..repo import MemoryRepo
from .utils import make_commit
class CanFastForwardTests(TestCase):

    def test_ff(self):
        r = MemoryRepo()
        base = make_commit()
        c1 = make_commit(parents=[base.id])
        c2 = make_commit(parents=[c1.id])
        r.object_store.add_objects([(base, None), (c1, None), (c2, None)])
        self.assertTrue(can_fast_forward(r, c1.id, c1.id))
        self.assertTrue(can_fast_forward(r, base.id, c1.id))
        self.assertTrue(can_fast_forward(r, c1.id, c2.id))
        self.assertFalse(can_fast_forward(r, c2.id, c1.id))

    def test_diverged(self):
        r = MemoryRepo()
        base = make_commit()
        c1 = make_commit(parents=[base.id])
        c2a = make_commit(parents=[c1.id], message=b'2a')
        c2b = make_commit(parents=[c1.id], message=b'2b')
        r.object_store.add_objects([(base, None), (c1, None), (c2a, None), (c2b, None)])
        self.assertTrue(can_fast_forward(r, c1.id, c2a.id))
        self.assertTrue(can_fast_forward(r, c1.id, c2b.id))
        self.assertFalse(can_fast_forward(r, c2a.id, c2b.id))
        self.assertFalse(can_fast_forward(r, c2b.id, c2a.id))