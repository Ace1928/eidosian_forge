from breezy import conflicts, errors, osutils, revisiontree, tests
from breezy import transport as _mod_transport
from breezy.bzr import workingtree_4
from breezy.tests import TestSkipped
from breezy.tests.per_tree import TestCaseWithTree
from breezy.tree import MissingNestedTree
class TestSupportsVersionableKind(TestCaseWithTree):

    def test_file(self):
        work_tree = self.make_branch_and_tree('tree')
        tree = self._convert_tree(work_tree)
        self.assertTrue(tree.versionable_kind('file'))

    def test_unknown(self):
        work_tree = self.make_branch_and_tree('tree')
        tree = self._convert_tree(work_tree)
        self.assertFalse(tree.versionable_kind('unknown'))