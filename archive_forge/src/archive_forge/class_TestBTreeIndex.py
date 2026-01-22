import pprint
import zlib
from ... import errors, fifo_cache, lru_cache, osutils, tests, transport
from ...tests import TestCaseWithTransport, features, scenarios
from .. import btree_index
from .. import index as _mod_index
class TestBTreeIndex(BTreeTestCase):

    def make_index(self, ref_lists=0, key_elements=1, nodes=[]):
        builder = btree_index.BTreeBuilder(reference_lists=ref_lists, key_elements=key_elements)
        for key, value, references in nodes:
            builder.add_node(key, value, references)
        stream = builder.finish()
        trans = transport.get_transport_from_url('trace+' + self.get_url())
        size = trans.put_file('index', stream)
        return btree_index.BTreeGraphIndex(trans, 'index', size)

    def make_index_with_offset(self, ref_lists=1, key_elements=1, nodes=[], offset=0):
        builder = btree_index.BTreeBuilder(key_elements=key_elements, reference_lists=ref_lists)
        builder.add_nodes(nodes)
        transport = self.get_transport('')
        temp_file = builder.finish()
        content = temp_file.read()
        del temp_file
        size = len(content)
        transport.put_bytes('index', b' ' * offset + content)
        return btree_index.BTreeGraphIndex(transport, 'index', size=size, offset=offset)

    def test_clear_cache(self):
        nodes = self.make_nodes(160, 2, 2)
        index = self.make_index(ref_lists=2, key_elements=2, nodes=nodes)
        self.assertEqual(1, len(list(index.iter_entries([nodes[30][0]]))))
        self.assertEqual([1, 4], index._row_lengths)
        self.assertIsNot(None, index._root_node)
        internal_node_pre_clear = set(index._internal_node_cache)
        self.assertTrue(len(index._leaf_node_cache) > 0)
        index.clear_cache()
        self.assertIsNot(None, index._root_node)
        self.assertEqual(internal_node_pre_clear, set(index._internal_node_cache))
        self.assertEqual(0, len(index._leaf_node_cache))

    def test_trivial_constructor(self):
        t = transport.get_transport_from_url('trace+' + self.get_url(''))
        index = btree_index.BTreeGraphIndex(t, 'index', None)
        self.assertEqual([], t._activity)

    def test_with_size_constructor(self):
        t = transport.get_transport_from_url('trace+' + self.get_url(''))
        index = btree_index.BTreeGraphIndex(t, 'index', 1)
        self.assertEqual([], t._activity)

    def test_empty_key_count_no_size(self):
        builder = btree_index.BTreeBuilder(key_elements=1, reference_lists=0)
        t = transport.get_transport_from_url('trace+' + self.get_url(''))
        t.put_file('index', builder.finish())
        index = btree_index.BTreeGraphIndex(t, 'index', None)
        del t._activity[:]
        self.assertEqual([], t._activity)
        self.assertEqual(0, index.key_count())
        self.assertEqual([('get', 'index')], t._activity)

    def test_empty_key_count(self):
        builder = btree_index.BTreeBuilder(key_elements=1, reference_lists=0)
        t = transport.get_transport_from_url('trace+' + self.get_url(''))
        size = t.put_file('index', builder.finish())
        self.assertEqual(72, size)
        index = btree_index.BTreeGraphIndex(t, 'index', size)
        del t._activity[:]
        self.assertEqual([], t._activity)
        self.assertEqual(0, index.key_count())
        self.assertEqual([('readv', 'index', [(0, 72)], False, None)], t._activity)

    def test_non_empty_key_count_2_2(self):
        builder = btree_index.BTreeBuilder(key_elements=2, reference_lists=2)
        nodes = self.make_nodes(35, 2, 2)
        for node in nodes:
            builder.add_node(*node)
        t = transport.get_transport_from_url('trace+' + self.get_url(''))
        size = t.put_file('index', builder.finish())
        index = btree_index.BTreeGraphIndex(t, 'index', size)
        del t._activity[:]
        self.assertEqual([], t._activity)
        self.assertEqual(70, index.key_count())
        self.assertEqual([('readv', 'index', [(0, size)], False, None)], t._activity)
        self.assertEqualApproxCompressed(1173, size)

    def test_with_offset_no_size(self):
        index = self.make_index_with_offset(key_elements=1, ref_lists=1, offset=1234, nodes=self.make_nodes(200, 1, 1))
        index._size = None
        self.assertEqual(200, index.key_count())

    def test_with_small_offset(self):
        index = self.make_index_with_offset(key_elements=1, ref_lists=1, offset=1234, nodes=self.make_nodes(200, 1, 1))
        self.assertEqual(200, index.key_count())

    def test_with_large_offset(self):
        index = self.make_index_with_offset(key_elements=1, ref_lists=1, offset=123456, nodes=self.make_nodes(200, 1, 1))
        self.assertEqual(200, index.key_count())

    def test__read_nodes_no_size_one_page_reads_once(self):
        self.make_index(nodes=[((b'key',), b'value', ())])
        trans = transport.get_transport_from_url('trace+' + self.get_url())
        index = btree_index.BTreeGraphIndex(trans, 'index', None)
        del trans._activity[:]
        nodes = dict(index._read_nodes([0]))
        self.assertEqual({0}, set(nodes))
        node = nodes[0]
        self.assertEqual([(b'key',)], node.all_keys())
        self.assertEqual([('get', 'index')], trans._activity)

    def test__read_nodes_no_size_multiple_pages(self):
        index = self.make_index(2, 2, nodes=self.make_nodes(160, 2, 2))
        index.key_count()
        num_pages = index._row_offsets[-1]
        trans = transport.get_transport_from_url('trace+' + self.get_url())
        index = btree_index.BTreeGraphIndex(trans, 'index', None)
        del trans._activity[:]
        nodes = dict(index._read_nodes([0]))
        self.assertEqual(list(range(num_pages)), sorted(nodes))

    def test_2_levels_key_count_2_2(self):
        builder = btree_index.BTreeBuilder(key_elements=2, reference_lists=2)
        nodes = self.make_nodes(160, 2, 2)
        for node in nodes:
            builder.add_node(*node)
        t = transport.get_transport_from_url('trace+' + self.get_url(''))
        size = t.put_file('index', builder.finish())
        self.assertEqualApproxCompressed(17692, size)
        index = btree_index.BTreeGraphIndex(t, 'index', size)
        del t._activity[:]
        self.assertEqual([], t._activity)
        self.assertEqual(320, index.key_count())
        self.assertEqual([('readv', 'index', [(0, 4096)], False, None)], t._activity)

    def test_validate_one_page(self):
        builder = btree_index.BTreeBuilder(key_elements=2, reference_lists=2)
        nodes = self.make_nodes(45, 2, 2)
        for node in nodes:
            builder.add_node(*node)
        t = transport.get_transport_from_url('trace+' + self.get_url(''))
        size = t.put_file('index', builder.finish())
        index = btree_index.BTreeGraphIndex(t, 'index', size)
        del t._activity[:]
        self.assertEqual([], t._activity)
        index.validate()
        self.assertEqual([('readv', 'index', [(0, size)], False, None)], t._activity)
        self.assertEqualApproxCompressed(1488, size)

    def test_validate_two_pages(self):
        builder = btree_index.BTreeBuilder(key_elements=2, reference_lists=2)
        nodes = self.make_nodes(80, 2, 2)
        for node in nodes:
            builder.add_node(*node)
        t = transport.get_transport_from_url('trace+' + self.get_url(''))
        size = t.put_file('index', builder.finish())
        self.assertEqualApproxCompressed(9339, size)
        index = btree_index.BTreeGraphIndex(t, 'index', size)
        del t._activity[:]
        self.assertEqual([], t._activity)
        index.validate()
        rem = size - 8192
        self.assertEqual([('readv', 'index', [(0, 4096)], False, None), ('readv', 'index', [(4096, 4096), (8192, rem)], False, None)], t._activity)

    def test_eq_ne(self):
        t1 = transport.get_transport_from_url('trace+' + self.get_url(''))
        t2 = self.get_transport()
        self.assertTrue(btree_index.BTreeGraphIndex(t1, 'index', None) == btree_index.BTreeGraphIndex(t1, 'index', None))
        self.assertTrue(btree_index.BTreeGraphIndex(t1, 'index', 20) == btree_index.BTreeGraphIndex(t1, 'index', 20))
        self.assertFalse(btree_index.BTreeGraphIndex(t1, 'index', 20) == btree_index.BTreeGraphIndex(t2, 'index', 20))
        self.assertFalse(btree_index.BTreeGraphIndex(t1, 'inde1', 20) == btree_index.BTreeGraphIndex(t1, 'inde2', 20))
        self.assertFalse(btree_index.BTreeGraphIndex(t1, 'index', 10) == btree_index.BTreeGraphIndex(t1, 'index', 20))
        self.assertFalse(btree_index.BTreeGraphIndex(t1, 'index', None) != btree_index.BTreeGraphIndex(t1, 'index', None))
        self.assertFalse(btree_index.BTreeGraphIndex(t1, 'index', 20) != btree_index.BTreeGraphIndex(t1, 'index', 20))
        self.assertTrue(btree_index.BTreeGraphIndex(t1, 'index', 20) != btree_index.BTreeGraphIndex(t2, 'index', 20))
        self.assertTrue(btree_index.BTreeGraphIndex(t1, 'inde1', 20) != btree_index.BTreeGraphIndex(t1, 'inde2', 20))
        self.assertTrue(btree_index.BTreeGraphIndex(t1, 'index', 10) != btree_index.BTreeGraphIndex(t1, 'index', 20))

    def test_key_too_big(self):
        bigKey = b''.join((b'%d' % n for n in range(btree_index._PAGE_SIZE)))
        self.assertRaises(_mod_index.BadIndexKey, self.make_index, nodes=[((bigKey,), b'value', ())])

    def test_iter_all_only_root_no_size(self):
        self.make_index(nodes=[((b'key',), b'value', ())])
        t = transport.get_transport_from_url('trace+' + self.get_url(''))
        index = btree_index.BTreeGraphIndex(t, 'index', None)
        del t._activity[:]
        self.assertEqual([((b'key',), b'value')], [x[1:] for x in index.iter_all_entries()])
        self.assertEqual([('get', 'index')], t._activity)

    def test_iter_all_entries_reads(self):
        self.shrink_page_size()
        builder = btree_index.BTreeBuilder(key_elements=2, reference_lists=2)
        nodes = self.make_nodes(10000, 2, 2)
        for node in nodes:
            builder.add_node(*node)
        t = transport.get_transport_from_url('trace+' + self.get_url(''))
        size = t.put_file('index', builder.finish())
        page_size = btree_index._PAGE_SIZE
        del builder
        index = btree_index.BTreeGraphIndex(t, 'index', size)
        del t._activity[:]
        self.assertEqual([], t._activity)
        found_nodes = self.time(list, index.iter_all_entries())
        bare_nodes = []
        for node in found_nodes:
            self.assertTrue(node[0] is index)
            bare_nodes.append(node[1:])
        self.assertEqual(3, len(index._row_lengths), 'Not enough rows: %r' % index._row_lengths)
        self.assertEqual(20000, len(found_nodes))
        self.assertEqual(set(nodes), set(bare_nodes))
        total_pages = sum(index._row_lengths)
        self.assertEqual(total_pages, index._row_offsets[-1])
        self.assertEqualApproxCompressed(1303220, size)
        first_byte = index._row_offsets[-2] * page_size
        readv_request = []
        for offset in range(first_byte, size, page_size):
            readv_request.append((offset, page_size))
        readv_request[-1] = (readv_request[-1][0], size % page_size)
        expected = [('readv', 'index', [(0, page_size)], False, None), ('readv', 'index', readv_request, False, None)]
        if expected != t._activity:
            self.assertEqualDiff(pprint.pformat(expected), pprint.pformat(t._activity))

    def test_iter_entries_references_2_refs_resolved(self):
        builder = btree_index.BTreeBuilder(key_elements=2, reference_lists=2)
        nodes = self.make_nodes(160, 2, 2)
        for node in nodes:
            builder.add_node(*node)
        t = transport.get_transport_from_url('trace+' + self.get_url(''))
        size = t.put_file('index', builder.finish())
        del builder
        index = btree_index.BTreeGraphIndex(t, 'index', size)
        del t._activity[:]
        self.assertEqual([], t._activity)
        found_nodes = list(index.iter_entries([nodes[30][0]]))
        bare_nodes = []
        for node in found_nodes:
            self.assertTrue(node[0] is index)
            bare_nodes.append(node[1:])
        self.assertEqual(1, len(found_nodes))
        self.assertEqual(nodes[30], bare_nodes[0])
        self.assertEqual([('readv', 'index', [(0, 4096)], False, None), ('readv', 'index', [(8192, 4096)], False, None)], t._activity)

    def test_iter_key_prefix_1_element_key_None(self):
        index = self.make_index()
        self.assertRaises(_mod_index.BadIndexKey, list, index.iter_entries_prefix([(None,)]))

    def test_iter_key_prefix_wrong_length(self):
        index = self.make_index()
        self.assertRaises(_mod_index.BadIndexKey, list, index.iter_entries_prefix([(b'foo', None)]))
        index = self.make_index(key_elements=2)
        self.assertRaises(_mod_index.BadIndexKey, list, index.iter_entries_prefix([(b'foo',)]))
        self.assertRaises(_mod_index.BadIndexKey, list, index.iter_entries_prefix([(b'foo', None, None)]))

    def test_iter_key_prefix_1_key_element_no_refs(self):
        index = self.make_index(nodes=[((b'name',), b'data', ()), ((b'ref',), b'refdata', ())])
        self.assertEqual({(index, (b'name',), b'data'), (index, (b'ref',), b'refdata')}, set(index.iter_entries_prefix([(b'name',), (b'ref',)])))

    def test_iter_key_prefix_1_key_element_refs(self):
        index = self.make_index(1, nodes=[((b'name',), b'data', ([(b'ref',)],)), ((b'ref',), b'refdata', ([],))])
        self.assertEqual({(index, (b'name',), b'data', (((b'ref',),),)), (index, (b'ref',), b'refdata', ((),))}, set(index.iter_entries_prefix([(b'name',), (b'ref',)])))

    def test_iter_key_prefix_2_key_element_no_refs(self):
        index = self.make_index(key_elements=2, nodes=[((b'name', b'fin1'), b'data', ()), ((b'name', b'fin2'), b'beta', ()), ((b'ref', b'erence'), b'refdata', ())])
        self.assertEqual({(index, (b'name', b'fin1'), b'data'), (index, (b'ref', b'erence'), b'refdata')}, set(index.iter_entries_prefix([(b'name', b'fin1'), (b'ref', b'erence')])))
        self.assertEqual({(index, (b'name', b'fin1'), b'data'), (index, (b'name', b'fin2'), b'beta')}, set(index.iter_entries_prefix([(b'name', None)])))

    def test_iter_key_prefix_2_key_element_refs(self):
        index = self.make_index(1, key_elements=2, nodes=[((b'name', b'fin1'), b'data', ([(b'ref', b'erence')],)), ((b'name', b'fin2'), b'beta', ([],)), ((b'ref', b'erence'), b'refdata', ([],))])
        self.assertEqual({(index, (b'name', b'fin1'), b'data', (((b'ref', b'erence'),),)), (index, (b'ref', b'erence'), b'refdata', ((),))}, set(index.iter_entries_prefix([(b'name', b'fin1'), (b'ref', b'erence')])))
        self.assertEqual({(index, (b'name', b'fin1'), b'data', (((b'ref', b'erence'),),)), (index, (b'name', b'fin2'), b'beta', ((),))}, set(index.iter_entries_prefix([(b'name', None)])))

    def test_external_references_no_refs(self):
        index = self.make_index(ref_lists=0, nodes=[])
        self.assertRaises(ValueError, index.external_references, 0)

    def test_external_references_no_results(self):
        index = self.make_index(ref_lists=1, nodes=[((b'key',), b'value', ([],))])
        self.assertEqual(set(), index.external_references(0))

    def test_external_references_missing_ref(self):
        missing_key = (b'missing',)
        index = self.make_index(ref_lists=1, nodes=[((b'key',), b'value', ([missing_key],))])
        self.assertEqual({missing_key}, index.external_references(0))

    def test_external_references_multiple_ref_lists(self):
        missing_key = (b'missing',)
        index = self.make_index(ref_lists=2, nodes=[((b'key',), b'value', ([], [missing_key]))])
        self.assertEqual(set(), index.external_references(0))
        self.assertEqual({missing_key}, index.external_references(1))

    def test_external_references_two_records(self):
        index = self.make_index(ref_lists=1, nodes=[((b'key-1',), b'value', ([(b'key-2',)],)), ((b'key-2',), b'value', ([],))])
        self.assertEqual(set(), index.external_references(0))

    def test__find_ancestors_one_page(self):
        key1 = (b'key-1',)
        key2 = (b'key-2',)
        index = self.make_index(ref_lists=1, key_elements=1, nodes=[(key1, b'value', ([key2],)), (key2, b'value', ([],))])
        parent_map = {}
        missing_keys = set()
        search_keys = index._find_ancestors([key1], 0, parent_map, missing_keys)
        self.assertEqual({key1: (key2,), key2: ()}, parent_map)
        self.assertEqual(set(), missing_keys)
        self.assertEqual(set(), search_keys)

    def test__find_ancestors_one_page_w_missing(self):
        key1 = (b'key-1',)
        key2 = (b'key-2',)
        key3 = (b'key-3',)
        index = self.make_index(ref_lists=1, key_elements=1, nodes=[(key1, b'value', ([key2],)), (key2, b'value', ([],))])
        parent_map = {}
        missing_keys = set()
        search_keys = index._find_ancestors([key2, key3], 0, parent_map, missing_keys)
        self.assertEqual({key2: ()}, parent_map)
        self.assertEqual({key3}, missing_keys)
        self.assertEqual(set(), search_keys)

    def test__find_ancestors_one_parent_missing(self):
        key1 = (b'key-1',)
        key2 = (b'key-2',)
        key3 = (b'key-3',)
        index = self.make_index(ref_lists=1, key_elements=1, nodes=[(key1, b'value', ([key2],)), (key2, b'value', ([key3],))])
        parent_map = {}
        missing_keys = set()
        search_keys = index._find_ancestors([key1], 0, parent_map, missing_keys)
        self.assertEqual({key1: (key2,), key2: (key3,)}, parent_map)
        self.assertEqual(set(), missing_keys)
        self.assertEqual({key3}, search_keys)
        search_keys = index._find_ancestors(search_keys, 0, parent_map, missing_keys)
        self.assertEqual({key1: (key2,), key2: (key3,)}, parent_map)
        self.assertEqual({key3}, missing_keys)
        self.assertEqual(set(), search_keys)

    def test__find_ancestors_dont_search_known(self):
        key1 = (b'key-1',)
        key2 = (b'key-2',)
        key3 = (b'key-3',)
        index = self.make_index(ref_lists=1, key_elements=1, nodes=[(key1, b'value', ([key2],)), (key2, b'value', ([key3],)), (key3, b'value', ([],))])
        parent_map = {key2: (key3,)}
        missing_keys = set()
        search_keys = index._find_ancestors([key1], 0, parent_map, missing_keys)
        self.assertEqual({key1: (key2,), key2: (key3,)}, parent_map)
        self.assertEqual(set(), missing_keys)
        self.assertEqual(set(), search_keys)

    def test__find_ancestors_multiple_pages(self):
        start_time = 1249671539
        email = 'joebob@example.com'
        nodes = []
        ref_lists = ((),)
        rev_keys = []
        for i in range(400):
            rev_id = '{}-{}-{}'.format(email, osutils.compact_date(start_time + i), osutils.rand_chars(16)).encode('ascii')
            rev_key = (rev_id,)
            nodes.append((rev_key, b'value', ref_lists))
            ref_lists = ((rev_key,),)
            rev_keys.append(rev_key)
        index = self.make_index(ref_lists=1, key_elements=1, nodes=nodes)
        self.assertEqual(400, index.key_count())
        self.assertEqual(3, len(index._row_offsets))
        nodes = dict(index._read_nodes([1, 2]))
        l1 = nodes[1]
        l2 = nodes[2]
        min_l2_key = l2.min_key
        max_l1_key = l1.max_key
        self.assertTrue(max_l1_key < min_l2_key)
        parents_min_l2_key = l2[min_l2_key][1][0]
        self.assertEqual((l1.max_key,), parents_min_l2_key)
        key_idx = rev_keys.index(min_l2_key)
        next_key = rev_keys[key_idx + 1]
        parent_map = {}
        missing_keys = set()
        search_keys = index._find_ancestors([next_key], 0, parent_map, missing_keys)
        self.assertEqual([min_l2_key, next_key], sorted(parent_map))
        self.assertEqual(set(), missing_keys)
        self.assertEqual({max_l1_key}, search_keys)
        parent_map = {}
        search_keys = index._find_ancestors([max_l1_key], 0, parent_map, missing_keys)
        self.assertEqual(l1.all_keys(), sorted(parent_map))
        self.assertEqual(set(), missing_keys)
        self.assertEqual(set(), search_keys)

    def test__find_ancestors_empty_index(self):
        index = self.make_index(ref_lists=1, key_elements=1, nodes=[])
        parent_map = {}
        missing_keys = set()
        search_keys = index._find_ancestors([('one',), ('two',)], 0, parent_map, missing_keys)
        self.assertEqual(set(), search_keys)
        self.assertEqual({}, parent_map)
        self.assertEqual({('one',), ('two',)}, missing_keys)

    def test_supports_unlimited_cache(self):
        builder = btree_index.BTreeBuilder(reference_lists=0, key_elements=1)
        nodes = self.make_nodes(500, 1, 0)
        for node in nodes:
            builder.add_node(*node)
        stream = builder.finish()
        trans = self.get_transport()
        size = trans.put_file('index', stream)
        index = btree_index.BTreeGraphIndex(trans, 'index', size)
        self.assertEqual(500, index.key_count())
        self.assertEqual(2, len(index._row_lengths))
        self.assertTrue(index._row_lengths[-1] >= 2)
        self.assertIsInstance(index._leaf_node_cache, lru_cache.LRUCache)
        self.assertEqual(btree_index._NODE_CACHE_SIZE, index._leaf_node_cache._max_cache)
        self.assertIsInstance(index._internal_node_cache, fifo_cache.FIFOCache)
        self.assertEqual(100, index._internal_node_cache._max_cache)
        index = btree_index.BTreeGraphIndex(trans, 'index', size, unlimited_cache=False)
        self.assertIsInstance(index._leaf_node_cache, lru_cache.LRUCache)
        self.assertEqual(btree_index._NODE_CACHE_SIZE, index._leaf_node_cache._max_cache)
        self.assertIsInstance(index._internal_node_cache, fifo_cache.FIFOCache)
        self.assertEqual(100, index._internal_node_cache._max_cache)
        index = btree_index.BTreeGraphIndex(trans, 'index', size, unlimited_cache=True)
        self.assertIsInstance(index._leaf_node_cache, dict)
        self.assertIs(type(index._internal_node_cache), dict)
        entries = set(index.iter_entries([n[0] for n in nodes]))
        self.assertEqual(500, len(entries))