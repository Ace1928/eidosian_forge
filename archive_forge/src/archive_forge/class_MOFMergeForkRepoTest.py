from dulwich.tests import TestCase
from ..object_store import MemoryObjectStore, MissingObjectFinder
from ..objects import Blob
from .utils import build_commit_graph, make_object, make_tag
class MOFMergeForkRepoTest(MissingObjectFinderTest):

    def setUp(self):
        super().setUp()
        f1_1 = make_object(Blob, data=b'f1')
        f1_2 = make_object(Blob, data=b'f1-2')
        f1_4 = make_object(Blob, data=b'f1-4')
        f1_7 = make_object(Blob, data=b'f1-2')
        f2_1 = make_object(Blob, data=b'f2')
        f2_3 = make_object(Blob, data=b'f2-3')
        f3_3 = make_object(Blob, data=b'f3')
        f3_5 = make_object(Blob, data=b'f3-5')
        commit_spec = [[1], [2, 1], [3, 2], [4, 2], [5, 3], [6, 3, 4], [7, 6]]
        trees = {1: [(b'f1', f1_1), (b'f2', f2_1)], 2: [(b'f1', f1_2), (b'f2', f2_1)], 3: [(b'f1', f1_2), (b'f2', f2_3), (b'f3', f3_3)], 4: [(b'f1', f1_4), (b'f2', f2_1)], 5: [(b'f1', f1_2), (b'f3', f3_5)], 6: [(b'f1', f1_4), (b'f2', f2_3), (b'f3', f3_3)], 7: [(b'f1', f1_7), (b'f2', f2_3)]}
        self.commits = build_commit_graph(self.store, commit_spec, trees)
        self.f1_2_id = f1_2.id
        self.f1_4_id = f1_4.id
        self.f1_7_id = f1_7.id
        self.f2_3_id = f2_3.id
        self.f3_3_id = f3_3.id
        self.assertEqual(f1_2.id, f1_7.id, '[sanity]')

    def test_have6_want7(self):
        self.assertMissingMatch([self.cmt(6).id], [self.cmt(7).id], [self.cmt(7).id, self.cmt(7).tree, self.f1_7_id])

    def test_have4_want7(self):
        self.assertMissingMatch([self.cmt(4).id], [self.cmt(7).id], [self.cmt(7).id, self.cmt(6).id, self.cmt(3).id, self.cmt(7).tree, self.cmt(6).tree, self.cmt(3).tree, self.f2_3_id, self.f3_3_id])

    def test_have1_want6(self):
        self.assertMissingMatch([self.cmt(1).id], [self.cmt(6).id], [self.cmt(6).id, self.cmt(4).id, self.cmt(3).id, self.cmt(2).id, self.cmt(6).tree, self.cmt(4).tree, self.cmt(3).tree, self.cmt(2).tree, self.f1_2_id, self.f1_4_id, self.f2_3_id, self.f3_3_id])

    def test_have3_want6(self):
        self.assertMissingMatch([self.cmt(3).id], [self.cmt(7).id], [self.cmt(7).id, self.cmt(6).id, self.cmt(4).id, self.cmt(7).tree, self.cmt(6).tree, self.cmt(4).tree, self.f1_4_id])

    def test_have5_want7(self):
        self.assertMissingMatch([self.cmt(5).id], [self.cmt(7).id], [self.cmt(7).id, self.cmt(6).id, self.cmt(4).id, self.cmt(7).tree, self.cmt(6).tree, self.cmt(4).tree, self.f1_4_id])