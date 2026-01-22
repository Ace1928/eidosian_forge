from ....conflicts import ConflictList
from ....errors import ConflictsInTree, UnknownFormatError
from ....graph import DictParentsProvider, Graph
from ....revision import NULL_REVISION
from ....tests import TestCase, TestCaseWithTransport
from ....tests.matchers import RevisionHistoryMatches
from ....transport import NoSuchFile
from ..rebase import (REBASE_CURRENT_REVID_FILENAME, REBASE_PLAN_FILENAME,
class PlanCreatorTests(TestCaseWithTransport):

    def test_simple_plan_creator(self):
        wt = self.make_branch_and_tree('.')
        b = wt.branch
        with open('hello', 'w') as f:
            f.write('hello world')
        wt.add('hello')
        wt.commit(message='add hello', rev_id=b'bla')
        with open('hello', 'w') as f:
            f.write('world')
        wt.commit(message='change hello', rev_id=b'bloe')
        wt.set_last_revision(b'bla')
        b.generate_revision_history(b'bla')
        with open('hello', 'w') as f:
            f.write('world')
        wt.commit(message='change hello', rev_id=b'bla2')
        b.repository.lock_read()
        graph = b.repository.get_graph()
        self.assertEqual({b'bla2': (b'newbla2', (b'bloe',))}, generate_simple_plan(graph.find_difference(b.last_revision(), b'bla')[0], b'bla2', None, b'bloe', graph, lambda y, _: b'new' + y))
        b.repository.unlock()

    def test_simple_plan_creator_extra_history(self):
        wt = self.make_branch_and_tree('.')
        b = wt.branch
        with open('hello', 'w') as f:
            f.write('hello world')
        wt.add('hello')
        wt.commit(message='add hello', rev_id=b'bla')
        with open('hello', 'w') as f:
            f.write('world')
        wt.commit(message='change hello', rev_id=b'bloe')
        wt.set_last_revision(b'bla')
        b.generate_revision_history(b'bla')
        with open('hello', 'w') as f:
            f.write('world')
        wt.commit(message='change hello', rev_id=b'bla2')
        with open('hello', 'w') as f:
            f.write('universe')
        wt.commit(message='change hello again', rev_id=b'bla3')
        with b.repository.lock_read():
            graph = b.repository.get_graph()
            self.assertEqual({b'bla2': (b'newbla2', (b'bloe',)), b'bla3': (b'newbla3', (b'newbla2',))}, generate_simple_plan(graph.find_difference(b.last_revision(), b'bloe')[0], b'bla2', None, b'bloe', graph, lambda y, _: b'new' + y))

    def test_generate_transpose_plan(self):
        wt = self.make_branch_and_tree('.')
        b = wt.branch
        with open('hello', 'w') as f:
            f.write('hello world')
        wt.add('hello')
        wt.commit(message='add hello', rev_id=b'bla')
        with open('hello', 'w') as f:
            f.write('world')
        wt.commit(message='change hello', rev_id=b'bloe')
        wt.set_last_revision(b'bla')
        b.generate_revision_history(b'bla')
        with open('hello', 'w') as f:
            f.write('world')
        wt.commit(message='change hello', rev_id=b'bla2')
        with open('hello', 'w') as f:
            f.write('universe')
        wt.commit(message='change hello again', rev_id=b'bla3')
        wt.set_last_revision(b'bla')
        b.generate_revision_history(b'bla')
        with open('hello', 'w') as f:
            f.write('somebar')
        wt.commit(message='change hello yet again', rev_id=b'blie')
        wt.set_last_revision(NULL_REVISION)
        b.generate_revision_history(NULL_REVISION)
        wt.add('hello')
        wt.commit(message='add hello', rev_id=b'lala')
        b.repository.lock_read()
        graph = b.repository.get_graph()
        self.assertEqual({b'blie': (b'newblie', (b'lala',))}, generate_transpose_plan(graph.iter_ancestry([b'blie']), {b'bla': b'lala'}, graph, lambda y, _: b'new' + y))
        self.assertEqual({b'bla2': (b'newbla2', (b'lala',)), b'bla3': (b'newbla3', (b'newbla2',)), b'blie': (b'newblie', (b'lala',)), b'bloe': (b'newbloe', (b'lala',))}, generate_transpose_plan(graph.iter_ancestry(b.repository._all_revision_ids()), {b'bla': b'lala'}, graph, lambda y, _: b'new' + y))
        b.repository.unlock()

    def test_generate_transpose_plan_one(self):
        graph = Graph(DictParentsProvider({'bla': ('bloe',), 'bloe': (), 'lala': ()}))
        self.assertEqual({'bla': ('newbla', ('lala',))}, generate_transpose_plan(graph.iter_ancestry(['bla', 'bloe']), {'bloe': 'lala'}, graph, lambda y, _: 'new' + y))

    def test_plan_with_already_merged(self):
        """We need to use a merge base that makes sense.

        A
        | \\
        B  D
        | \\|
        C  E

        Rebasing E on C should result in:

        A -> B -> C -> D -> E

        with a plan of:

        D -> (D', [C])
        E -> (E', [D', C])
        """
        parents_map = {'A': (), 'B': ('A',), 'C': ('B',), 'D': ('A',), 'E': ('D', 'B')}
        graph = Graph(DictParentsProvider(parents_map))
        self.assertEqual({'D': ("D'", ('C',)), 'E': ("E'", ("D'",))}, generate_simple_plan(['D', 'E'], 'D', None, 'C', graph, lambda y, _: y + "'"))

    def test_plan_with_already_merged_skip_merges(self):
        """We need to use a merge base that makes sense.

        A
        | \\
        B  D
        | \\|
        C  E

        Rebasing E on C should result in:

        A -> B -> C -> D'

        with a plan of:

        D -> (D', [C])
        """
        parents_map = {'A': (), 'B': ('A',), 'C': ('B',), 'D': ('A',), 'E': ('D', 'B')}
        graph = Graph(DictParentsProvider(parents_map))
        self.assertEqual({'D': ("D'", ('C',))}, generate_simple_plan(['D', 'E'], 'D', None, 'C', graph, lambda y, _: y + "'", True))