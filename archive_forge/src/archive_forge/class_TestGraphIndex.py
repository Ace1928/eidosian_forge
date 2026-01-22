from ... import errors, tests, transport
from .. import index as _mod_index
class TestGraphIndex(tests.TestCaseWithMemoryTransport):

    def make_key(self, number):
        return (b'%d' % number + b'X' * 100,)

    def make_value(self, number):
        return b'%d' % number + b'Y' * 100

    def make_nodes(self, count=64):
        nodes = []
        for counter in range(count):
            nodes.append((self.make_key(counter), self.make_value(counter), ()))
        return nodes

    def make_index(self, ref_lists=0, key_elements=1, nodes=[]):
        builder = _mod_index.GraphIndexBuilder(ref_lists, key_elements=key_elements)
        for key, value, references in nodes:
            builder.add_node(key, value, references)
        stream = builder.finish()
        trans = transport.get_transport_from_url('trace+' + self.get_url())
        size = trans.put_file('index', stream)
        return _mod_index.GraphIndex(trans, 'index', size)

    def make_index_with_offset(self, ref_lists=0, key_elements=1, nodes=[], offset=0):
        builder = _mod_index.GraphIndexBuilder(ref_lists, key_elements=key_elements)
        for key, value, references in nodes:
            builder.add_node(key, value, references)
        content = builder.finish().read()
        size = len(content)
        trans = self.get_transport()
        trans.put_bytes('index', b' ' * offset + content)
        return _mod_index.GraphIndex(trans, 'index', size, offset=offset)

    def test_clear_cache(self):
        index = self.make_index()
        index.clear_cache()

    def test_open_bad_index_no_error(self):
        trans = self.get_transport()
        trans.put_bytes('name', b'not an index\n')
        idx = _mod_index.GraphIndex(trans, 'name', 13)

    def test_with_offset(self):
        nodes = self.make_nodes(200)
        idx = self.make_index_with_offset(offset=1234567, nodes=nodes)
        self.assertEqual(200, idx.key_count())

    def test_buffer_all_with_offset(self):
        nodes = self.make_nodes(200)
        idx = self.make_index_with_offset(offset=1234567, nodes=nodes)
        idx._buffer_all()
        self.assertEqual(200, idx.key_count())

    def test_side_effect_buffering_with_offset(self):
        nodes = self.make_nodes(20)
        index = self.make_index_with_offset(offset=1234567, nodes=nodes)
        index._transport.recommended_page_size = lambda: 64 * 1024
        subset_nodes = [nodes[0][0], nodes[10][0], nodes[19][0]]
        entries = [n[1] for n in index.iter_entries(subset_nodes)]
        self.assertEqual(sorted(subset_nodes), sorted(entries))
        self.assertEqual(20, index.key_count())

    def test_open_sets_parsed_map_empty(self):
        index = self.make_index()
        self.assertEqual([], index._parsed_byte_map)
        self.assertEqual([], index._parsed_key_map)

    def test_key_count_buffers(self):
        index = self.make_index(nodes=self.make_nodes(2))
        del index._transport._activity[:]
        self.assertEqual(2, index.key_count())
        self.assertEqual([('readv', 'index', [(0, 200)], True, index._size)], index._transport._activity)
        self.assertIsNot(None, index._nodes)

    def test_lookup_key_via_location_buffers(self):
        index = self.make_index()
        del index._transport._activity[:]
        result = index._lookup_keys_via_location([(index._size // 2, (b'missing',))])
        self.assertEqual([('readv', 'index', [(30, 30), (0, 200)], True, 60)], index._transport._activity)
        self.assertEqual([((index._size // 2, (b'missing',)), False)], result)
        self.assertIsNot(None, index._nodes)
        self.assertEqual([], index._parsed_byte_map)

    def test_first_lookup_key_via_location(self):
        nodes = []
        index = self.make_index(nodes=self.make_nodes(64))
        del index._transport._activity[:]
        start_lookup = index._size // 2
        result = index._lookup_keys_via_location([(start_lookup, (b'40missing',))])
        self.assertEqual([('readv', 'index', [(start_lookup, 800), (0, 200)], True, index._size)], index._transport._activity)
        self.assertEqual([((start_lookup, (b'40missing',)), False)], result)
        self.assertIs(None, index._nodes)
        self.assertEqual([(0, 4008), (5046, 8996)], index._parsed_byte_map)
        self.assertEqual([((), self.make_key(26)), (self.make_key(31), self.make_key(48))], index._parsed_key_map)

    def test_parsing_non_adjacent_data_trims(self):
        index = self.make_index(nodes=self.make_nodes(64))
        result = index._lookup_keys_via_location([(index._size // 2, (b'40',))])
        self.assertEqual([((index._size // 2, (b'40',)), False)], result)
        self.assertEqual([(0, 4008), (5046, 8996)], index._parsed_byte_map)
        self.assertEqual([((), self.make_key(26)), (self.make_key(31), self.make_key(48))], index._parsed_key_map)

    def test_parsing_data_handles_parsed_contained_regions(self):
        index = self.make_index(nodes=self.make_nodes(128))
        result = index._lookup_keys_via_location([(index._size // 2, (b'40',))])
        self.assertEqual([(0, 4045), (11759, 15707)], index._parsed_byte_map)
        self.assertEqual([((), self.make_key(116)), (self.make_key(35), self.make_key(51))], index._parsed_key_map)
        result = index._lookup_keys_via_location([(11450, self.make_key(34)), (15707, self.make_key(52))])
        self.assertEqual([((11450, self.make_key(34)), (index, self.make_key(34), self.make_value(34))), ((15707, self.make_key(52)), (index, self.make_key(52), self.make_value(52)))], result)
        self.assertEqual([(0, 4045), (9889, 17993)], index._parsed_byte_map)

    def test_lookup_missing_key_answers_without_io_when_map_permits(self):
        index = self.make_index(nodes=self.make_nodes(64))
        result = index._lookup_keys_via_location([(index._size // 2, (b'40',))])
        self.assertEqual([(0, 4008), (5046, 8996)], index._parsed_byte_map)
        self.assertEqual([((), self.make_key(26)), (self.make_key(31), self.make_key(48))], index._parsed_key_map)
        del index._transport._activity[:]
        result = index._lookup_keys_via_location([(4000, (b'40',))])
        self.assertEqual([((4000, (b'40',)), False)], result)
        self.assertEqual([], index._transport._activity)

    def test_lookup_present_key_answers_without_io_when_map_permits(self):
        index = self.make_index(nodes=self.make_nodes(64))
        result = index._lookup_keys_via_location([(index._size // 2, (b'40',))])
        self.assertEqual([(0, 4008), (5046, 8996)], index._parsed_byte_map)
        self.assertEqual([((), self.make_key(26)), (self.make_key(31), self.make_key(48))], index._parsed_key_map)
        del index._transport._activity[:]
        result = index._lookup_keys_via_location([(4000, self.make_key(40))])
        self.assertEqual([((4000, self.make_key(40)), (index, self.make_key(40), self.make_value(40)))], result)
        self.assertEqual([], index._transport._activity)

    def test_lookup_key_below_probed_area(self):
        index = self.make_index(nodes=self.make_nodes(64))
        result = index._lookup_keys_via_location([(index._size // 2, (b'30',))])
        self.assertEqual([(0, 4008), (5046, 8996)], index._parsed_byte_map)
        self.assertEqual([((), self.make_key(26)), (self.make_key(31), self.make_key(48))], index._parsed_key_map)
        self.assertEqual([((index._size // 2, (b'30',)), -1)], result)

    def test_lookup_key_above_probed_area(self):
        index = self.make_index(nodes=self.make_nodes(64))
        result = index._lookup_keys_via_location([(index._size // 2, (b'50',))])
        self.assertEqual([(0, 4008), (5046, 8996)], index._parsed_byte_map)
        self.assertEqual([((), self.make_key(26)), (self.make_key(31), self.make_key(48))], index._parsed_key_map)
        self.assertEqual([((index._size // 2, (b'50',)), +1)], result)

    def test_lookup_key_resolves_references(self):
        nodes = []
        for counter in range(99):
            nodes.append((self.make_key(counter), self.make_value(counter), ((self.make_key(counter + 20),),)))
        index = self.make_index(ref_lists=1, nodes=nodes)
        index_size = index._size
        index_center = index_size // 2
        result = index._lookup_keys_via_location([(index_center, (b'40',))])
        self.assertEqual([(0, 4027), (10198, 14028)], index._parsed_byte_map)
        self.assertEqual([((), self.make_key(17)), (self.make_key(44), self.make_key(5))], index._parsed_key_map)
        self.assertEqual([('readv', 'index', [(index_center, 800), (0, 200)], True, index_size)], index._transport._activity)
        del index._transport._activity[:]
        result = index._lookup_keys_via_location([(11000, self.make_key(45))])
        self.assertEqual([((11000, self.make_key(45)), (index, self.make_key(45), self.make_value(45), ((self.make_key(65),),)))], result)
        self.assertEqual([('readv', 'index', [(15093, 800)], True, index_size)], index._transport._activity)

    def test_lookup_key_can_buffer_all(self):
        nodes = []
        for counter in range(64):
            nodes.append((self.make_key(counter), self.make_value(counter), ((self.make_key(counter + 20),),)))
        index = self.make_index(ref_lists=1, nodes=nodes)
        index_size = index._size
        index_center = index_size // 2
        result = index._lookup_keys_via_location([(index_center, (b'40',))])
        self.assertEqual([(0, 3890), (6444, 10274)], index._parsed_byte_map)
        self.assertEqual([((), self.make_key(25)), (self.make_key(37), self.make_key(52))], index._parsed_key_map)
        self.assertEqual([('readv', 'index', [(index_center, 800), (0, 200)], True, index_size)], index._transport._activity)
        del index._transport._activity[:]
        result = index._lookup_keys_via_location([(7000, self.make_key(40))])
        self.assertEqual([((7000, self.make_key(40)), (index, self.make_key(40), self.make_value(40), ((self.make_key(60),),)))], result)
        self.assertEqual([('get', 'index')], index._transport._activity)

    def test_iter_all_entries_empty(self):
        index = self.make_index()
        self.assertEqual([], list(index.iter_all_entries()))

    def test_iter_all_entries_simple(self):
        index = self.make_index(nodes=[((b'name',), b'data', ())])
        self.assertEqual([(index, (b'name',), b'data')], list(index.iter_all_entries()))

    def test_iter_all_entries_simple_2_elements(self):
        index = self.make_index(key_elements=2, nodes=[((b'name', b'surname'), b'data', ())])
        self.assertEqual([(index, (b'name', b'surname'), b'data')], list(index.iter_all_entries()))

    def test_iter_all_entries_references_resolved(self):
        index = self.make_index(1, nodes=[((b'name',), b'data', ([(b'ref',)],)), ((b'ref',), b'refdata', ([],))])
        self.assertEqual({(index, (b'name',), b'data', (((b'ref',),),)), (index, (b'ref',), b'refdata', ((),))}, set(index.iter_all_entries()))

    def test_iter_entries_buffers_once(self):
        index = self.make_index(nodes=self.make_nodes(2))
        del index._transport._activity[:]
        self.assertEqual({(index, self.make_key(1), self.make_value(1))}, set(index.iter_entries([self.make_key(1)])))
        self.assertEqual([('readv', 'index', [(0, 200)], True, index._size)], index._transport._activity)
        self.assertIsNot(None, index._nodes)

    def test_iter_entries_buffers_by_bytes_read(self):
        index = self.make_index(nodes=self.make_nodes(64))
        list(index.iter_entries([self.make_key(10)]))
        self.assertIs(None, index._nodes)
        self.assertEqual(4096, index._bytes_read)
        list(index.iter_entries([self.make_key(11)]))
        self.assertIs(None, index._nodes)
        self.assertEqual(4096, index._bytes_read)
        list(index.iter_entries([self.make_key(40)]))
        self.assertIs(None, index._nodes)
        self.assertEqual(8192, index._bytes_read)
        list(index.iter_entries([self.make_key(32)]))
        self.assertIs(None, index._nodes)
        list(index.iter_entries([self.make_key(60)]))
        self.assertIsNot(None, index._nodes)

    def test_iter_entries_references_resolved(self):
        index = self.make_index(1, nodes=[((b'name',), b'data', ([(b'ref',), (b'ref',)],)), ((b'ref',), b'refdata', ([],))])
        self.assertEqual({(index, (b'name',), b'data', (((b'ref',), (b'ref',)),)), (index, (b'ref',), b'refdata', ((),))}, set(index.iter_entries([(b'name',), (b'ref',)])))

    def test_iter_entries_references_2_refs_resolved(self):
        index = self.make_index(2, nodes=[((b'name',), b'data', ([(b'ref',)], [(b'ref',)])), ((b'ref',), b'refdata', ([], []))])
        self.assertEqual({(index, (b'name',), b'data', (((b'ref',),), ((b'ref',),))), (index, (b'ref',), b'refdata', ((), ()))}, set(index.iter_entries([(b'name',), (b'ref',)])))

    def test_iteration_absent_skipped(self):
        index = self.make_index(1, nodes=[((b'name',), b'data', ([(b'ref',)],))])
        self.assertEqual({(index, (b'name',), b'data', (((b'ref',),),))}, set(index.iter_all_entries()))
        self.assertEqual({(index, (b'name',), b'data', (((b'ref',),),))}, set(index.iter_entries([(b'name',)])))
        self.assertEqual([], list(index.iter_entries([(b'ref',)])))

    def test_iteration_absent_skipped_2_element_keys(self):
        index = self.make_index(1, key_elements=2, nodes=[((b'name', b'fin'), b'data', ([(b'ref', b'erence')],))])
        self.assertEqual([(index, (b'name', b'fin'), b'data', (((b'ref', b'erence'),),))], list(index.iter_all_entries()))
        self.assertEqual([(index, (b'name', b'fin'), b'data', (((b'ref', b'erence'),),))], list(index.iter_entries([(b'name', b'fin')])))
        self.assertEqual([], list(index.iter_entries([(b'ref', b'erence')])))

    def test_iter_all_keys(self):
        index = self.make_index(1, nodes=[((b'name',), b'data', ([(b'ref',)],)), ((b'ref',), b'refdata', ([],))])
        self.assertEqual({(index, (b'name',), b'data', (((b'ref',),),)), (index, (b'ref',), b'refdata', ((),))}, set(index.iter_entries([(b'name',), (b'ref',)])))

    def test_iter_nothing_empty(self):
        index = self.make_index()
        self.assertEqual([], list(index.iter_entries([])))

    def test_iter_missing_entry_empty(self):
        index = self.make_index()
        self.assertEqual([], list(index.iter_entries([(b'a',)])))

    def test_iter_missing_entry_empty_no_size(self):
        idx = self.make_index()
        idx = _mod_index.GraphIndex(idx._transport, 'index', None)
        self.assertEqual([], list(idx.iter_entries([(b'a',)])))

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

    def test_key_count_empty(self):
        index = self.make_index()
        self.assertEqual(0, index.key_count())

    def test_key_count_one(self):
        index = self.make_index(nodes=[((b'name',), b'', ())])
        self.assertEqual(1, index.key_count())

    def test_key_count_two(self):
        index = self.make_index(nodes=[((b'name',), b'', ()), ((b'foo',), b'', ())])
        self.assertEqual(2, index.key_count())

    def test_read_and_parse_tracks_real_read_value(self):
        index = self.make_index(nodes=self.make_nodes(10))
        del index._transport._activity[:]
        index._read_and_parse([(0, 200)])
        self.assertEqual([('readv', 'index', [(0, 200)], True, index._size)], index._transport._activity)
        self.assertEqual(index._size, index._bytes_read)

    def test_read_and_parse_triggers_buffer_all(self):
        index = self.make_index(key_elements=2, nodes=[((b'name', b'fin1'), b'data', ()), ((b'name', b'fin2'), b'beta', ()), ((b'ref', b'erence'), b'refdata', ())])
        self.assertTrue(index._size > 0)
        self.assertIs(None, index._nodes)
        index._read_and_parse([(0, index._size)])
        self.assertIsNot(None, index._nodes)

    def test_validate_bad_index_errors(self):
        trans = self.get_transport()
        trans.put_bytes('name', b'not an index\n')
        idx = _mod_index.GraphIndex(trans, 'name', 13)
        self.assertRaises(_mod_index.BadIndexFormatSignature, idx.validate)

    def test_validate_bad_node_refs(self):
        idx = self.make_index(2)
        trans = self.get_transport()
        content = trans.get_bytes('index')
        new_content = content[:-2] + b'a\n\n'
        trans.put_bytes('index', new_content)
        self.assertRaises(_mod_index.BadIndexOptions, idx.validate)

    def test_validate_missing_end_line_empty(self):
        index = self.make_index(2)
        trans = self.get_transport()
        content = trans.get_bytes('index')
        trans.put_bytes('index', content[:-1])
        self.assertRaises(_mod_index.BadIndexData, index.validate)

    def test_validate_missing_end_line_nonempty(self):
        index = self.make_index(2, nodes=[((b'key',), b'', ([], []))])
        trans = self.get_transport()
        content = trans.get_bytes('index')
        trans.put_bytes('index', content[:-1])
        self.assertRaises(_mod_index.BadIndexData, index.validate)

    def test_validate_empty(self):
        index = self.make_index()
        index.validate()

    def test_validate_no_refs_content(self):
        index = self.make_index(nodes=[((b'key',), b'value', ())])
        index.validate()

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

    def test__find_ancestors(self):
        key1 = (b'key-1',)
        key2 = (b'key-2',)
        index = self.make_index(ref_lists=1, key_elements=1, nodes=[(key1, b'value', ([key2],)), (key2, b'value', ([],))])
        parent_map = {}
        missing_keys = set()
        search_keys = index._find_ancestors([key1], 0, parent_map, missing_keys)
        self.assertEqual({key1: (key2,)}, parent_map)
        self.assertEqual(set(), missing_keys)
        self.assertEqual({key2}, search_keys)
        search_keys = index._find_ancestors(search_keys, 0, parent_map, missing_keys)
        self.assertEqual({key1: (key2,), key2: ()}, parent_map)
        self.assertEqual(set(), missing_keys)
        self.assertEqual(set(), search_keys)

    def test__find_ancestors_w_missing(self):
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

    def test_supports_unlimited_cache(self):
        builder = _mod_index.GraphIndexBuilder(0, key_elements=1)
        stream = builder.finish()
        trans = self.get_transport()
        size = trans.put_file('index', stream)
        idx = _mod_index.GraphIndex(trans, 'index', size, unlimited_cache=True)