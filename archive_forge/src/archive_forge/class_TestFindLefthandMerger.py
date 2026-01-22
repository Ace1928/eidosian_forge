from .. import errors
from .. import graph as _mod_graph
from .. import tests
from ..revision import NULL_REVISION
from . import TestCaseWithMemoryTransport
class TestFindLefthandMerger(TestGraphBase):

    def check_merger(self, result, ancestry, merged, tip):
        graph = self.make_graph(ancestry)
        self.assertEqual(result, graph.find_lefthand_merger(merged, tip))

    def test_find_lefthand_merger_rev2b(self):
        self.check_merger(b'rev4', ancestry_1, b'rev2b', b'rev4')

    def test_find_lefthand_merger_rev2a(self):
        self.check_merger(b'rev2a', ancestry_1, b'rev2a', b'rev4')

    def test_find_lefthand_merger_rev4(self):
        self.check_merger(None, ancestry_1, b'rev4', b'rev2a')

    def test_find_lefthand_merger_f(self):
        self.check_merger(b'i', complex_shortcut, b'f', b'm')

    def test_find_lefthand_merger_g(self):
        self.check_merger(b'i', complex_shortcut, b'g', b'm')

    def test_find_lefthand_merger_h(self):
        self.check_merger(b'n', complex_shortcut, b'h', b'n')