from breezy import errors, gpg, tests, urlutils
from breezy.bzr.testament import Testament
from breezy.repository import WriteGroup
from breezy.tests import per_repository
def test_verify_revision_signature_not_signed(self):
    wt = self.make_branch_and_tree('.')
    a = wt.commit('base', allow_pointless=True)
    strategy = gpg.LoopbackGPGStrategy(None)
    self.assertEqual((gpg.SIGNATURE_NOT_SIGNED, None), wt.branch.repository.verify_revision_signature(a, strategy))