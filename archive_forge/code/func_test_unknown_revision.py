from breezy import errors
from breezy.tests.per_repository_reference import \
def test_unknown_revision(self):
    tree2 = self.make_branch_and_tree('other')
    unknown_revid = tree2.commit('other')
    repo = self.tree.branch.repository
    repo.lock_read()
    self.addCleanup(repo.unlock)
    self.assertRaises(errors.NoSuchRevision, repo.get_rev_id_for_revno, 1, (3, unknown_revid))