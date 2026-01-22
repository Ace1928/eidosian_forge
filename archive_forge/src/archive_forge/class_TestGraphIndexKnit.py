import gzip
import sys
from io import BytesIO
from patiencediff import PatienceSequenceMatcher
from ... import errors, multiparent, osutils, tests
from ... import transport as _mod_transport
from ...tests import (TestCase, TestCaseWithMemoryTransport,
from .. import knit, knitpack_repo, pack, pack_repo
from ..index import *
from ..knit import (AnnotatedKnitContent, KnitContent, KnitCorrupt,
from ..versionedfile import (AbsentContentFactory, ConstantMapper,
class TestGraphIndexKnit(KnitTests):
    """Tests for knits using a GraphIndex rather than a KnitIndex."""

    def make_g_index(self, name, ref_lists=0, nodes=[]):
        builder = GraphIndexBuilder(ref_lists)
        for node, references, value in nodes:
            builder.add_node(node, references, value)
        stream = builder.finish()
        trans = self.get_transport()
        size = trans.put_file(name, stream)
        return GraphIndex(trans, name, size)

    def two_graph_index(self, deltas=False, catch_adds=False):
        """Build a two-graph index.

        :param deltas: If true, use underlying indices with two node-ref
            lists and 'parent' set to a delta-compressed against tail.
        """
        if deltas:
            index1 = self.make_g_index('1', 2, [((b'tip',), b'N0 100', ([(b'parent',)], [])), ((b'tail',), b'', ([], []))])
            index2 = self.make_g_index('2', 2, [((b'parent',), b' 100 78', ([(b'tail',), (b'ghost',)], [(b'tail',)])), ((b'separate',), b'', ([], []))])
        else:
            index1 = self.make_g_index('1', 1, [((b'tip',), b'N0 100', ([(b'parent',)],)), ((b'tail',), b'', ([],))])
            index2 = self.make_g_index('2', 1, [((b'parent',), b' 100 78', ([(b'tail',), (b'ghost',)],)), ((b'separate',), b'', ([],))])
        combined_index = CombinedGraphIndex([index1, index2])
        if catch_adds:
            self.combined_index = combined_index
            self.caught_entries = []
            add_callback = self.catch_add
        else:
            add_callback = None
        return _KnitGraphIndex(combined_index, lambda: True, deltas=deltas, add_callback=add_callback)

    def test_keys(self):
        index = self.two_graph_index()
        self.assertEqual({(b'tail',), (b'tip',), (b'parent',), (b'separate',)}, set(index.keys()))

    def test_get_position(self):
        index = self.two_graph_index()
        self.assertEqual((index._graph_index._indices[0], 0, 100), index.get_position((b'tip',)))
        self.assertEqual((index._graph_index._indices[1], 100, 78), index.get_position((b'parent',)))

    def test_get_method_deltas(self):
        index = self.two_graph_index(deltas=True)
        self.assertEqual('fulltext', index.get_method((b'tip',)))
        self.assertEqual('line-delta', index.get_method((b'parent',)))

    def test_get_method_no_deltas(self):
        index = self.two_graph_index(deltas=False)
        self.assertEqual('fulltext', index.get_method((b'tip',)))
        self.assertEqual('fulltext', index.get_method((b'parent',)))

    def test_get_options_deltas(self):
        index = self.two_graph_index(deltas=True)
        self.assertEqual([b'fulltext', b'no-eol'], index.get_options((b'tip',)))
        self.assertEqual([b'line-delta'], index.get_options((b'parent',)))

    def test_get_options_no_deltas(self):
        index = self.two_graph_index(deltas=False)
        self.assertEqual([b'fulltext', b'no-eol'], index.get_options((b'tip',)))
        self.assertEqual([b'fulltext'], index.get_options((b'parent',)))

    def test_get_parent_map(self):
        index = self.two_graph_index()
        self.assertEqual({(b'parent',): ((b'tail',), (b'ghost',))}, index.get_parent_map([(b'parent',), (b'ghost',)]))

    def catch_add(self, entries):
        self.caught_entries.append(entries)

    def test_add_no_callback_errors(self):
        index = self.two_graph_index()
        self.assertRaises(errors.ReadOnlyError, index.add_records, [((b'new',), b'fulltext,no-eol', (None, 50, 60), [b'separate'])])

    def test_add_version_smoke(self):
        index = self.two_graph_index(catch_adds=True)
        index.add_records([((b'new',), b'fulltext,no-eol', (None, 50, 60), [(b'separate',)])])
        self.assertEqual([[((b'new',), b'N50 60', (((b'separate',),),))]], self.caught_entries)

    def test_add_version_delta_not_delta_index(self):
        index = self.two_graph_index(catch_adds=True)
        self.assertRaises(KnitCorrupt, index.add_records, [((b'new',), b'no-eol,line-delta', (None, 0, 100), [(b'parent',)])])
        self.assertEqual([], self.caught_entries)

    def test_add_version_same_dup(self):
        index = self.two_graph_index(catch_adds=True)
        index.add_records([((b'tip',), b'fulltext,no-eol', (None, 0, 100), [(b'parent',)])])
        index.add_records([((b'tip',), b'no-eol,fulltext', (None, 0, 100), [(b'parent',)])])
        index.add_records([((b'tip',), b'fulltext,no-eol', (None, 50, 100), [(b'parent',)])])
        index.add_records([((b'tip',), b'fulltext,no-eol', (None, 0, 1000), [(b'parent',)])])
        self.assertEqual([[], [], [], []], self.caught_entries)

    def test_add_version_different_dup(self):
        index = self.two_graph_index(deltas=True, catch_adds=True)
        self.assertRaises(KnitCorrupt, index.add_records, [((b'tip',), b'line-delta', (None, 0, 100), [(b'parent',)])])
        self.assertRaises(KnitCorrupt, index.add_records, [((b'tip',), b'fulltext', (None, 0, 100), [(b'parent',)])])
        self.assertRaises(KnitCorrupt, index.add_records, [((b'tip',), b'fulltext,no-eol', (None, 0, 100), [])])
        self.assertEqual([], self.caught_entries)

    def test_add_versions_nodeltas(self):
        index = self.two_graph_index(catch_adds=True)
        index.add_records([((b'new',), b'fulltext,no-eol', (None, 50, 60), [(b'separate',)]), ((b'new2',), b'fulltext', (None, 0, 6), [(b'new',)])])
        self.assertEqual([((b'new',), b'N50 60', (((b'separate',),),)), ((b'new2',), b' 0 6', (((b'new',),),))], sorted(self.caught_entries[0]))
        self.assertEqual(1, len(self.caught_entries))

    def test_add_versions_deltas(self):
        index = self.two_graph_index(deltas=True, catch_adds=True)
        index.add_records([((b'new',), b'fulltext,no-eol', (None, 50, 60), [(b'separate',)]), ((b'new2',), b'line-delta', (None, 0, 6), [(b'new',)])])
        self.assertEqual([((b'new',), b'N50 60', (((b'separate',),), ())), ((b'new2',), b' 0 6', (((b'new',),), ((b'new',),)))], sorted(self.caught_entries[0]))
        self.assertEqual(1, len(self.caught_entries))

    def test_add_versions_delta_not_delta_index(self):
        index = self.two_graph_index(catch_adds=True)
        self.assertRaises(KnitCorrupt, index.add_records, [((b'new',), b'no-eol,line-delta', (None, 0, 100), [(b'parent',)])])
        self.assertEqual([], self.caught_entries)

    def test_add_versions_random_id_accepted(self):
        index = self.two_graph_index(catch_adds=True)
        index.add_records([], random_id=True)

    def test_add_versions_same_dup(self):
        index = self.two_graph_index(catch_adds=True)
        index.add_records([((b'tip',), b'fulltext,no-eol', (None, 0, 100), [(b'parent',)])])
        index.add_records([((b'tip',), b'no-eol,fulltext', (None, 0, 100), [(b'parent',)])])
        index.add_records([((b'tip',), b'fulltext,no-eol', (None, 50, 100), [(b'parent',)])])
        index.add_records([((b'tip',), b'fulltext,no-eol', (None, 0, 1000), [(b'parent',)])])
        self.assertEqual([[], [], [], []], self.caught_entries)

    def test_add_versions_different_dup(self):
        index = self.two_graph_index(deltas=True, catch_adds=True)
        self.assertRaises(KnitCorrupt, index.add_records, [((b'tip',), b'line-delta', (None, 0, 100), [(b'parent',)])])
        self.assertRaises(KnitCorrupt, index.add_records, [((b'tip',), b'fulltext', (None, 0, 100), [(b'parent',)])])
        self.assertRaises(KnitCorrupt, index.add_records, [((b'tip',), b'fulltext,no-eol', (None, 0, 100), [])])
        self.assertRaises(KnitCorrupt, index.add_records, [((b'tip',), b'fulltext,no-eol', (None, 0, 100), [(b'parent',)]), ((b'tip',), b'line-delta', (None, 0, 100), [(b'parent',)])])
        self.assertEqual([], self.caught_entries)

    def make_g_index_missing_compression_parent(self):
        graph_index = self.make_g_index('missing_comp', 2, [((b'tip',), b' 100 78', ([(b'missing-parent',), (b'ghost',)], [(b'missing-parent',)]))])
        return graph_index

    def make_g_index_missing_parent(self):
        graph_index = self.make_g_index('missing_parent', 2, [((b'parent',), b' 100 78', ([], [])), ((b'tip',), b' 100 78', ([(b'parent',), (b'missing-parent',)], [(b'parent',)]))])
        return graph_index

    def make_g_index_no_external_refs(self):
        graph_index = self.make_g_index('no_external_refs', 2, [((b'rev',), b' 100 78', ([(b'parent',), (b'ghost',)], []))])
        return graph_index

    def test_add_good_unvalidated_index(self):
        unvalidated = self.make_g_index_no_external_refs()
        combined = CombinedGraphIndex([unvalidated])
        index = _KnitGraphIndex(combined, lambda: True, deltas=True)
        index.scan_unvalidated_index(unvalidated)
        self.assertEqual(frozenset(), index.get_missing_compression_parents())

    def test_add_missing_compression_parent_unvalidated_index(self):
        unvalidated = self.make_g_index_missing_compression_parent()
        combined = CombinedGraphIndex([unvalidated])
        index = _KnitGraphIndex(combined, lambda: True, deltas=True)
        index.scan_unvalidated_index(unvalidated)
        self.assertEqual(frozenset([(b'missing-parent',)]), index.get_missing_compression_parents())

    def test_add_missing_noncompression_parent_unvalidated_index(self):
        unvalidated = self.make_g_index_missing_parent()
        combined = CombinedGraphIndex([unvalidated])
        index = _KnitGraphIndex(combined, lambda: True, deltas=True, track_external_parent_refs=True)
        index.scan_unvalidated_index(unvalidated)
        self.assertEqual(frozenset([(b'missing-parent',)]), index.get_missing_parents())

    def test_track_external_parent_refs(self):
        g_index = self.make_g_index('empty', 2, [])
        combined = CombinedGraphIndex([g_index])
        index = _KnitGraphIndex(combined, lambda: True, deltas=True, add_callback=self.catch_add, track_external_parent_refs=True)
        self.caught_entries = []
        index.add_records([((b'new-key',), b'fulltext,no-eol', (None, 50, 60), [(b'parent-1',), (b'parent-2',)])])
        self.assertEqual(frozenset([(b'parent-1',), (b'parent-2',)]), index.get_missing_parents())

    def test_add_unvalidated_index_with_present_external_references(self):
        index = self.two_graph_index(deltas=True)
        unvalidated = index._graph_index._indices[1]
        index.scan_unvalidated_index(unvalidated)
        self.assertEqual(frozenset(), index.get_missing_compression_parents())

    def make_new_missing_parent_g_index(self, name):
        missing_parent = name.encode('ascii') + b'-missing-parent'
        graph_index = self.make_g_index(name, 2, [((name.encode('ascii') + b'tip',), b' 100 78', ([(missing_parent,), (b'ghost',)], [(missing_parent,)]))])
        return graph_index

    def test_add_mulitiple_unvalidated_indices_with_missing_parents(self):
        g_index_1 = self.make_new_missing_parent_g_index('one')
        g_index_2 = self.make_new_missing_parent_g_index('two')
        combined = CombinedGraphIndex([g_index_1, g_index_2])
        index = _KnitGraphIndex(combined, lambda: True, deltas=True)
        index.scan_unvalidated_index(g_index_1)
        index.scan_unvalidated_index(g_index_2)
        self.assertEqual(frozenset([(b'one-missing-parent',), (b'two-missing-parent',)]), index.get_missing_compression_parents())

    def test_add_mulitiple_unvalidated_indices_with_mutual_dependencies(self):
        graph_index_a = self.make_g_index('one', 2, [((b'parent-one',), b' 100 78', ([(b'non-compression-parent',)], [])), ((b'child-of-two',), b' 100 78', ([(b'parent-two',)], [(b'parent-two',)]))])
        graph_index_b = self.make_g_index('two', 2, [((b'parent-two',), b' 100 78', ([(b'non-compression-parent',)], [])), ((b'child-of-one',), b' 100 78', ([(b'parent-one',)], [(b'parent-one',)]))])
        combined = CombinedGraphIndex([graph_index_a, graph_index_b])
        index = _KnitGraphIndex(combined, lambda: True, deltas=True)
        index.scan_unvalidated_index(graph_index_a)
        index.scan_unvalidated_index(graph_index_b)
        self.assertEqual(frozenset([]), index.get_missing_compression_parents())