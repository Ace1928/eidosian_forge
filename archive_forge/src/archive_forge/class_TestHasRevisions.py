from breezy.revision import NULL_REVISION
from breezy.tests.per_repository import TestCaseWithRepository
class TestHasRevisions(TestCaseWithRepository):

    def test_empty_list(self):
        repo = self.make_repository('.')
        self.assertEqual(set(), repo.has_revisions([]))

    def test_superset(self):
        tree = self.make_branch_and_tree('.')
        repo = tree.branch.repository
        rev1 = tree.commit('1')
        rev2 = tree.commit('2')
        rev3 = tree.commit('3')
        self.assertEqual({rev1, rev3}, repo.has_revisions([rev1, rev3, b'foobar:']))

    def test_NULL(self):
        repo = self.make_repository('.')
        self.assertEqual({NULL_REVISION}, repo.has_revisions([NULL_REVISION]))