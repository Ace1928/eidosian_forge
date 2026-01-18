import zlib
from ... import config, errors, osutils, tests, trace
from ...osutils import sha_string
from ...tests.scenarios import load_tests_apply_scenarios
from .. import btree_index, groupcompress
from .. import index as _mod_index
from .. import knit, versionedfile
from .test__groupcompress import compiled_groupcompress_feature
def test_yield_factories_flushing(self):
    """yield_factories holds back on yielding results from the final block
        unless passed full_flush=True.
        """
    fake_block = groupcompress.GroupCompressBlock()
    read_memo = ('fake index', 100, 50)
    gcvf = StubGCVF()
    gcvf._group_cache[read_memo] = fake_block
    locations = {('key',): (read_memo + (0, 0), None, None, None)}
    batcher = groupcompress._BatchingBlockFetcher(gcvf, locations)
    batcher.add_key(('key',))
    self.assertEqual([], list(batcher.yield_factories()))
    factories = list(batcher.yield_factories(full_flush=True))
    self.assertLength(1, factories)
    self.assertEqual(('key',), factories[0].key)
    self.assertEqual('groupcompress-block', factories[0].storage_kind)