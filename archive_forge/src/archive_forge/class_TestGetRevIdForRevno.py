from breezy import errors
from breezy.tests.per_repository_reference import \
class TestGetRevIdForRevno(TestCaseWithExternalReferenceRepository):

    def test_uses_fallback(self):
        tree = self.make_branch_and_tree('base')
        base = tree.branch.repository
        revid = tree.commit('one')
        revid2 = tree.commit('two')
        spare_tree = tree.controldir.sprout('spare').open_workingtree()
        revid3 = spare_tree.commit('three')
        branch = spare_tree.branch.create_clone_on_transport(self.get_transport('referring'), stacked_on=tree.branch.base)
        repo = branch.repository
        self.assertEqual({revid3}, set(repo.controldir.open_repository().all_revision_ids()))
        self.assertEqual({revid2, revid}, set(base.controldir.open_repository().all_revision_ids()))
        repo.lock_read()
        self.addCleanup(repo.unlock)
        self.assertEqual((True, revid), repo.get_rev_id_for_revno(1, (3, revid3)))