from breezy import conflicts, errors, osutils, revisiontree, tests
from breezy import transport as _mod_transport
from breezy.bzr import workingtree_4
from breezy.tests import TestSkipped
from breezy.tests.per_tree import TestCaseWithTree
from breezy.tree import MissingNestedTree
def test_stored_kind(self):
    tree = self.make_branch_and_tree('tree')
    work_tree = self.make_branch_and_tree('wt')
    tree = self.get_tree_no_parents_abc_content(work_tree)
    tree.lock_read()
    self.addCleanup(tree.unlock)
    self.assertEqual('file', tree.stored_kind('a'))
    self.assertEqual('directory', tree.stored_kind('b'))