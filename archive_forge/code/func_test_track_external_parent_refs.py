import zlib
from ... import config, errors, osutils, tests, trace
from ...osutils import sha_string
from ...tests.scenarios import load_tests_apply_scenarios
from .. import btree_index, groupcompress
from .. import index as _mod_index
from .. import knit, versionedfile
from .test__groupcompress import compiled_groupcompress_feature
def test_track_external_parent_refs(self):
    g_index = self.make_g_index('empty', 1, [])
    mod_index = btree_index.BTreeBuilder(1, 1)
    combined = _mod_index.CombinedGraphIndex([g_index, mod_index])
    index = groupcompress._GCGraphIndex(combined, is_locked=lambda: True, parents=True, add_callback=mod_index.add_nodes, track_external_parent_refs=True)
    index.add_records([((b'new-key',), b'2 10 2 10', [((b'parent-1',), (b'parent-2',))])])
    self.assertEqual(frozenset([(b'parent-1',), (b'parent-2',)]), index.get_missing_parents())