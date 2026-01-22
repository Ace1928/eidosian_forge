from ... import errors, tests, transport
from .. import index as _mod_index
class TestInMemoryGraphIndex(tests.TestCaseWithMemoryTransport):

    def make_index(self, ref_lists=0, key_elements=1, nodes=[]):
        result = _mod_index.InMemoryGraphIndex(ref_lists, key_elements=key_elements)
        result.add_nodes(nodes)
        return result

    def test_add_nodes_no_refs(self):
        index = self.make_index(0)
        index.add_nodes([((b'name',), b'data')])
        index.add_nodes([((b'name2',), b''), ((b'name3',), b'')])
        self.assertEqual({(index, (b'name',), b'data'), (index, (b'name2',), b''), (index, (b'name3',), b'')}, set(index.iter_all_entries()))

    def test_add_nodes(self):
        index = self.make_index(1)
        index.add_nodes([((b'name',), b'data', ([],))])
        index.add_nodes([((b'name2',), b'', ([],)), ((b'name3',), b'', ([(b'r',)],))])
        self.assertEqual({(index, (b'name',), b'data', ((),)), (index, (b'name2',), b'', ((),)), (index, (b'name3',), b'', (((b'r',),),))}, set(index.iter_all_entries()))

    def test_iter_all_entries_empty(self):
        index = self.make_index()
        self.assertEqual([], list(index.iter_all_entries()))

    def test_iter_all_entries_simple(self):
        index = self.make_index(nodes=[((b'name',), b'data')])
        self.assertEqual([(index, (b'name',), b'data')], list(index.iter_all_entries()))

    def test_iter_all_entries_references(self):
        index = self.make_index(1, nodes=[((b'name',), b'data', ([(b'ref',)],)), ((b'ref',), b'refdata', ([],))])
        self.assertEqual({(index, (b'name',), b'data', (((b'ref',),),)), (index, (b'ref',), b'refdata', ((),))}, set(index.iter_all_entries()))

    def test_iteration_absent_skipped(self):
        index = self.make_index(1, nodes=[((b'name',), b'data', ([(b'ref',)],))])
        self.assertEqual({(index, (b'name',), b'data', (((b'ref',),),))}, set(index.iter_all_entries()))
        self.assertEqual({(index, (b'name',), b'data', (((b'ref',),),))}, set(index.iter_entries([(b'name',)])))
        self.assertEqual([], list(index.iter_entries([(b'ref',)])))

    def test_iter_all_keys(self):
        index = self.make_index(1, nodes=[((b'name',), b'data', ([(b'ref',)],)), ((b'ref',), b'refdata', ([],))])
        self.assertEqual({(index, (b'name',), b'data', (((b'ref',),),)), (index, (b'ref',), b'refdata', ((),))}, set(index.iter_entries([(b'name',), (b'ref',)])))

    def test_iter_key_prefix_1_key_element_no_refs(self):
        index = self.make_index(nodes=[((b'name',), b'data'), ((b'ref',), b'refdata')])
        self.assertEqual({(index, (b'name',), b'data'), (index, (b'ref',), b'refdata')}, set(index.iter_entries_prefix([(b'name',), (b'ref',)])))

    def test_iter_key_prefix_1_key_element_refs(self):
        index = self.make_index(1, nodes=[((b'name',), b'data', ([(b'ref',)],)), ((b'ref',), b'refdata', ([],))])
        self.assertEqual({(index, (b'name',), b'data', (((b'ref',),),)), (index, (b'ref',), b'refdata', ((),))}, set(index.iter_entries_prefix([(b'name',), (b'ref',)])))

    def test_iter_key_prefix_2_key_element_no_refs(self):
        index = self.make_index(key_elements=2, nodes=[((b'name', b'fin1'), b'data'), ((b'name', b'fin2'), b'beta'), ((b'ref', b'erence'), b'refdata')])
        self.assertEqual({(index, (b'name', b'fin1'), b'data'), (index, (b'ref', b'erence'), b'refdata')}, set(index.iter_entries_prefix([(b'name', b'fin1'), (b'ref', b'erence')])))
        self.assertEqual({(index, (b'name', b'fin1'), b'data'), (index, (b'name', b'fin2'), b'beta')}, set(index.iter_entries_prefix([(b'name', None)])))

    def test_iter_key_prefix_2_key_element_refs(self):
        index = self.make_index(1, key_elements=2, nodes=[((b'name', b'fin1'), b'data', ([(b'ref', b'erence')],)), ((b'name', b'fin2'), b'beta', ([],)), ((b'ref', b'erence'), b'refdata', ([],))])
        self.assertEqual({(index, (b'name', b'fin1'), b'data', (((b'ref', b'erence'),),)), (index, (b'ref', b'erence'), b'refdata', ((),))}, set(index.iter_entries_prefix([(b'name', b'fin1'), (b'ref', b'erence')])))
        self.assertEqual({(index, (b'name', b'fin1'), b'data', (((b'ref', b'erence'),),)), (index, (b'name', b'fin2'), b'beta', ((),))}, set(index.iter_entries_prefix([(b'name', None)])))

    def test_iter_nothing_empty(self):
        index = self.make_index()
        self.assertEqual([], list(index.iter_entries([])))

    def test_iter_missing_entry_empty(self):
        index = self.make_index()
        self.assertEqual([], list(index.iter_entries([b'a'])))

    def test_key_count_empty(self):
        index = self.make_index()
        self.assertEqual(0, index.key_count())

    def test_key_count_one(self):
        index = self.make_index(nodes=[((b'name',), b'')])
        self.assertEqual(1, index.key_count())

    def test_key_count_two(self):
        index = self.make_index(nodes=[((b'name',), b''), ((b'foo',), b'')])
        self.assertEqual(2, index.key_count())

    def test_validate_empty(self):
        index = self.make_index()
        index.validate()

    def test_validate_no_refs_content(self):
        index = self.make_index(nodes=[((b'key',), b'value')])
        index.validate()