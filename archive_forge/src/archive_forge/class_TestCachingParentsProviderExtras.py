from .. import errors
from .. import graph as _mod_graph
from .. import tests
from ..revision import NULL_REVISION
from . import TestCaseWithMemoryTransport
class TestCachingParentsProviderExtras(tests.TestCaseWithTransport):
    """Test the behaviour when parents are provided that were not requested."""

    def setUp(self):
        super().setUp()

        class ExtraParentsProvider:

            def get_parent_map(self, keys):
                return {b'rev1': [], b'rev2': [b'rev1']}
        self.inst_pp = InstrumentedParentsProvider(ExtraParentsProvider())
        self.caching_pp = _mod_graph.CachingParentsProvider(get_parent_map=self.inst_pp.get_parent_map)

    def test_uncached(self):
        self.caching_pp.disable_cache()
        self.assertEqual({b'rev1': []}, self.caching_pp.get_parent_map([b'rev1']))
        self.assertEqual([b'rev1'], self.inst_pp.calls)
        self.assertIs(None, self.caching_pp._cache)

    def test_cache_initially_empty(self):
        self.assertEqual({}, self.caching_pp._cache)

    def test_cached(self):
        self.assertEqual({b'rev1': []}, self.caching_pp.get_parent_map([b'rev1']))
        self.assertEqual([b'rev1'], self.inst_pp.calls)
        self.assertEqual({b'rev1': [], b'rev2': [b'rev1']}, self.caching_pp._cache)
        self.assertEqual({b'rev1': []}, self.caching_pp.get_parent_map([b'rev1']))
        self.assertEqual([b'rev1'], self.inst_pp.calls)

    def test_disable_cache_clears_cache(self):
        self.caching_pp.get_parent_map([b'rev1'])
        self.assertEqual(2, len(self.caching_pp._cache))
        self.caching_pp.disable_cache()
        self.assertIs(None, self.caching_pp._cache)

    def test_enable_cache_raises(self):
        e = self.assertRaises(AssertionError, self.caching_pp.enable_cache)
        self.assertEqual('Cache enabled when already enabled.', str(e))

    def test_cache_misses(self):
        self.caching_pp.get_parent_map([b'rev3'])
        self.caching_pp.get_parent_map([b'rev3'])
        self.assertEqual([b'rev3'], self.inst_pp.calls)

    def test_no_cache_misses(self):
        self.caching_pp.disable_cache()
        self.caching_pp.enable_cache(cache_misses=False)
        self.caching_pp.get_parent_map([b'rev3'])
        self.caching_pp.get_parent_map([b'rev3'])
        self.assertEqual([b'rev3', b'rev3'], self.inst_pp.calls)

    def test_cache_extras(self):
        self.assertEqual({}, self.caching_pp.get_parent_map([b'rev3']))
        self.assertEqual({b'rev2': [b'rev1']}, self.caching_pp.get_parent_map([b'rev2']))
        self.assertEqual([b'rev3'], self.inst_pp.calls)

    def test_extras_using_cached(self):
        self.assertEqual({}, self.caching_pp.get_cached_parent_map([b'rev3']))
        self.assertEqual({}, self.caching_pp.get_parent_map([b'rev3']))
        self.assertEqual({b'rev2': [b'rev1']}, self.caching_pp.get_cached_parent_map([b'rev2']))
        self.assertEqual([b'rev3'], self.inst_pp.calls)