from breezy import revision
from breezy.tests.per_repository import TestCaseWithRepository
class TestGetParentMap(TestCaseWithRepository):

    def test_missing_revision(self):
        tree = self.make_branch_and_tree('.')
        repo = tree.branch.repository
        repo.lock_read()
        self.addCleanup(repo.unlock)
        self.assertEqual({}, repo.get_parent_map([b'non-existant']))

    def test_multiple_parents(self):
        tree = self.make_branch_and_tree('.')
        rev1 = tree.commit('first')
        rev2 = tree.commit('second')
        tree.set_parent_ids([rev1, rev2])
        tree.branch.set_last_revision_info(1, rev1)
        rev3 = tree.commit('third')
        repo = tree.branch.repository
        repo.lock_read()
        self.addCleanup(repo.unlock)
        self.assertEqual({rev3: (rev1, rev2)}, repo.get_parent_map([rev3]))
        self.assertEqual({rev1: (revision.NULL_REVISION,), rev2: (rev1,), rev3: (rev1, rev2)}, repo.get_parent_map([rev1, rev2, rev3]))

    def test_no_parents(self):
        tree = self.make_branch_and_tree('.')
        rev1 = tree.commit('first')
        repo = tree.branch.repository
        repo.lock_read()
        self.addCleanup(repo.unlock)
        self.assertEqual({rev1: (revision.NULL_REVISION,)}, repo.get_parent_map([rev1]))

    def test_none(self):
        tree = self.make_branch_and_tree('.')
        rev1 = tree.commit('first')
        repo = tree.branch.repository
        repo.lock_read()
        self.addCleanup(repo.unlock)
        self.assertRaises(ValueError, repo.get_parent_map, [None])

    def test_null_revision(self):
        tree = self.make_branch_and_tree('.')
        repo = tree.branch.repository
        repo.lock_read()
        self.addCleanup(repo.unlock)
        self.assertEqual({revision.NULL_REVISION: ()}, repo.get_parent_map([revision.NULL_REVISION]))