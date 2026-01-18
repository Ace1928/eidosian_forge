from .. import errors
from .. import graph as _mod_graph
from .. import tests
from ..revision import NULL_REVISION
from . import TestCaseWithMemoryTransport
def test_note_missing_key(self):
    """After noting that a key is missing it is cached."""
    self.caching_pp.note_missing_key(b'b')
    self.assertEqual({}, self.caching_pp.get_parent_map([b'b']))
    self.assertEqual([], self.inst_pp.calls)
    self.assertEqual({b'b'}, self.caching_pp.missing_keys)