from breezy import errors, gpg, tests, urlutils
from breezy.bzr.testament import Testament
from breezy.repository import WriteGroup
from breezy.tests import per_repository
def test_sign_revision(self):
    if self.repository_format.supports_revision_signatures:
        raise tests.TestNotApplicable('repository supports signing revisions')
    wt = self.make_branch_and_tree('source')
    a = wt.commit('A', allow_pointless=True)
    repo = wt.branch.repository
    repo.lock_write()
    repo.start_write_group()
    self.assertRaises(errors.UnsupportedOperation, repo.sign_revision, a, gpg.LoopbackGPGStrategy(None))
    repo.commit_write_group()