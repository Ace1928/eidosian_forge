import socket
from breezy import revision
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_socket(self):
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.bind('tree/socketpath')
    s.listen(1)
    empty_tree = self.tree.branch.repository.revision_tree(revision.NULL_REVISION)
    d = self.tree.changes_from(empty_tree, specific_files=['socketpath'], want_unversioned=True)
    self.assertEqual([], d.added)
    self.assertEqual([], d.removed)
    self.assertEqual([], d.renamed)
    self.assertEqual([], d.copied)
    self.assertEqual([], d.modified)
    self.assertIn([x.path[1] for x in d.unversioned], [['socketpath'], []])