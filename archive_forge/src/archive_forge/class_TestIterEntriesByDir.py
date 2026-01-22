from breezy import conflicts, errors, osutils, revisiontree, tests
from breezy import transport as _mod_transport
from breezy.bzr import workingtree_4
from breezy.tests import TestSkipped
from breezy.tests.per_tree import TestCaseWithTree
from breezy.tree import MissingNestedTree
class TestIterEntriesByDir(TestCaseWithTree):

    def test_iteration_order(self):
        work_tree = self.make_branch_and_tree('.')
        self.build_tree(['a/', 'a/b/', 'a/b/c', 'a/d/', 'a/d/e', 'f/', 'f/g'])
        work_tree.add(['a', 'a/b', 'a/b/c', 'a/d', 'a/d/e', 'f', 'f/g'])
        tree = self._convert_tree(work_tree)
        output_order = [p for p, e in tree.iter_entries_by_dir()]
        self.assertEqual(['', 'a', 'f', 'a/b', 'a/d', 'a/b/c', 'a/d/e', 'f/g'], output_order)