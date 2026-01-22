import zlib
from ... import config, errors, osutils, tests, trace
from ...osutils import sha_string
from ...tests.scenarios import load_tests_apply_scenarios
from .. import btree_index, groupcompress
from .. import index as _mod_index
from .. import knit, versionedfile
from .test__groupcompress import compiled_groupcompress_feature
class Test_BatchingBlockFetcher(TestCaseWithGroupCompressVersionedFiles):
    """Simple whitebox unit tests for _BatchingBlockFetcher."""

    def test_add_key_new_read_memo(self):
        """Adding a key with an uncached read_memo new to this batch adds that
        read_memo to the list of memos to fetch.
        """
        read_memo = ('fake index', 100, 50)
        locations = {('key',): (read_memo + (None, None), None, None, None)}
        batcher = groupcompress._BatchingBlockFetcher(StubGCVF(), locations)
        total_size = batcher.add_key(('key',))
        self.assertEqual(50, total_size)
        self.assertEqual([('key',)], batcher.keys)
        self.assertEqual([read_memo], batcher.memos_to_get)

    def test_add_key_duplicate_read_memo(self):
        """read_memos that occur multiple times in a batch will only be fetched
        once.
        """
        read_memo = ('fake index', 100, 50)
        locations = {('key1',): (read_memo + (0, 1), None, None, None), ('key2',): (read_memo + (1, 2), None, None, None)}
        batcher = groupcompress._BatchingBlockFetcher(StubGCVF(), locations)
        total_size = batcher.add_key(('key1',))
        total_size = batcher.add_key(('key2',))
        self.assertEqual(50, total_size)
        self.assertEqual([('key1',), ('key2',)], batcher.keys)
        self.assertEqual([read_memo], batcher.memos_to_get)

    def test_add_key_cached_read_memo(self):
        """Adding a key with a cached read_memo will not cause that read_memo
        to be added to the list to fetch.
        """
        read_memo = ('fake index', 100, 50)
        gcvf = StubGCVF()
        gcvf._group_cache[read_memo] = 'fake block'
        locations = {('key',): (read_memo + (None, None), None, None, None)}
        batcher = groupcompress._BatchingBlockFetcher(gcvf, locations)
        total_size = batcher.add_key(('key',))
        self.assertEqual(0, total_size)
        self.assertEqual([('key',)], batcher.keys)
        self.assertEqual([], batcher.memos_to_get)

    def test_yield_factories_empty(self):
        """An empty batch yields no factories."""
        batcher = groupcompress._BatchingBlockFetcher(StubGCVF(), {})
        self.assertEqual([], list(batcher.yield_factories()))

    def test_yield_factories_calls_get_blocks(self):
        """Uncached memos are retrieved via get_blocks."""
        read_memo1 = ('fake index', 100, 50)
        read_memo2 = ('fake index', 150, 40)
        gcvf = StubGCVF(canned_get_blocks=[(read_memo1, groupcompress.GroupCompressBlock()), (read_memo2, groupcompress.GroupCompressBlock())])
        locations = {('key1',): (read_memo1 + (0, 0), None, None, None), ('key2',): (read_memo2 + (0, 0), None, None, None)}
        batcher = groupcompress._BatchingBlockFetcher(gcvf, locations)
        batcher.add_key(('key1',))
        batcher.add_key(('key2',))
        factories = list(batcher.yield_factories(full_flush=True))
        self.assertLength(2, factories)
        keys = [f.key for f in factories]
        kinds = [f.storage_kind for f in factories]
        self.assertEqual([('key1',), ('key2',)], keys)
        self.assertEqual(['groupcompress-block', 'groupcompress-block'], kinds)

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