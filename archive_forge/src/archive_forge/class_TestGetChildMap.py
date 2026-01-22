from .. import errors
from .. import graph as _mod_graph
from .. import tests
from ..revision import NULL_REVISION
from . import TestCaseWithMemoryTransport
class TestGetChildMap(TestGraphBase):

    def test_get_child_map(self):
        graph = self.make_graph(ancestry_1)
        child_map = graph.get_child_map([b'rev4', b'rev3', b'rev2a', b'rev2b'])
        self.assertEqual({b'rev1': [b'rev2a', b'rev2b'], b'rev2a': [b'rev3'], b'rev2b': [b'rev4'], b'rev3': [b'rev4']}, child_map)