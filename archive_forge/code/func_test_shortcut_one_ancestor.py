from .. import errors
from .. import graph as _mod_graph
from .. import tests
from ..revision import NULL_REVISION
from . import TestCaseWithMemoryTransport
def test_shortcut_one_ancestor(self):
    graph = self.make_breaking_graph(ancestry_1, [b'rev3', b'rev2b', b'rev4'])
    self.assertMergeOrder([b'rev3'], graph, b'rev4', [b'rev3'])