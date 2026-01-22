from breezy import revision
from breezy.tests import TestCaseWithTransport
from breezy.tree import FileTimestampUnavailable
class TestTreeWithCommits(TestCaseWithTransport):

    def setUp(self):
        super().setUp()
        self.t = self.make_branch_and_tree('.')
        self.rev_id = self.t.commit('foo', allow_pointless=True)
        self.rev_tree = self.t.branch.repository.revision_tree(self.rev_id)

    def test_empty_no_unknowns(self):
        self.assertEqual([], list(self.rev_tree.unknowns()))

    def test_no_conflicts(self):
        self.assertEqual([], list(self.rev_tree.conflicts()))

    def test_parents(self):
        """RevisionTree.parent_ids should match the revision graph."""
        self.assertEqual([], self.rev_tree.get_parent_ids())
        revid_2 = self.t.commit('bar', allow_pointless=True)
        self.assertEqual([self.rev_id], self.t.branch.repository.revision_tree(revid_2).get_parent_ids())
        self.assertEqual([], self.t.branch.repository.revision_tree(revision.NULL_REVISION).get_parent_ids())

    def test_empty_no_root(self):
        null_tree = self.t.branch.repository.revision_tree(revision.NULL_REVISION)
        self.assertIs(None, null_tree.path2id(''))

    def test_get_file_revision_root(self):
        self.assertEqual(self.rev_id, self.rev_tree.get_file_revision(''))

    def test_get_file_revision(self):
        self.build_tree_contents([('a', b'initial')])
        self.t.add(['a'])
        revid1 = self.t.commit('add a')
        revid2 = self.t.commit('another change', allow_pointless=True)
        tree = self.t.branch.repository.revision_tree(revid2)
        self.assertEqual(revid1, tree.get_file_revision('a'))

    def test_get_file_mtime_ghost(self):
        path = next(iter(self.rev_tree.all_versioned_paths()))
        self.rev_tree.root_inventory.get_entry(self.rev_tree.path2id(path)).revision = b'ghostrev'
        self.assertRaises(FileTimestampUnavailable, self.rev_tree.get_file_mtime, path)