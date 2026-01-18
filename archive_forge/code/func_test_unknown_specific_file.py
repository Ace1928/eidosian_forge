import socket
from breezy import revision
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_unknown_specific_file(self):
    self.build_tree(['tree/unknown'])
    empty_tree = self.tree.branch.repository.revision_tree(revision.NULL_REVISION)
    d = self.tree.changes_from(empty_tree, specific_files=['unknown'])
    self.assertEqual([], d.added)
    self.assertEqual([], d.removed)
    self.assertEqual([], d.renamed)
    self.assertEqual([], d.copied)
    self.assertEqual([], d.modified)