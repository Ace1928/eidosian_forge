from .. import errors
from .. import graph as _mod_graph
from .. import tests
from ..revision import NULL_REVISION
from . import TestCaseWithMemoryTransport
def test_uncached(self):
    self.caching_pp.disable_cache()
    self.assertEqual({b'rev1': []}, self.caching_pp.get_parent_map([b'rev1']))
    self.assertEqual([b'rev1'], self.inst_pp.calls)
    self.assertIs(None, self.caching_pp._cache)