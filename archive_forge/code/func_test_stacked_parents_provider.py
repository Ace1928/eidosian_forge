from .. import errors
from .. import graph as _mod_graph
from .. import tests
from ..revision import NULL_REVISION
from . import TestCaseWithMemoryTransport
def test_stacked_parents_provider(self):
    parents1 = _mod_graph.DictParentsProvider({b'rev2': [b'rev3']})
    parents2 = _mod_graph.DictParentsProvider({b'rev1': [b'rev4']})
    stacked = _mod_graph.StackedParentsProvider([parents1, parents2])
    self.assertEqual({b'rev1': [b'rev4'], b'rev2': [b'rev3']}, stacked.get_parent_map([b'rev1', b'rev2']))
    self.assertEqual({b'rev2': [b'rev3'], b'rev1': [b'rev4']}, stacked.get_parent_map([b'rev2', b'rev1']))
    self.assertEqual({b'rev2': [b'rev3']}, stacked.get_parent_map([b'rev2', b'rev2']))
    self.assertEqual({b'rev1': [b'rev4']}, stacked.get_parent_map([b'rev1', b'rev1']))