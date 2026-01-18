from breezy import errors, gpg, tests, urlutils
from breezy.bzr.testament import Testament
from breezy.repository import WriteGroup
from breezy.tests import per_repository
def test_store_signature(self):
    wt = self.make_branch_and_tree('.')
    branch = wt.branch
    with branch.lock_write(), WriteGroup(branch.repository):
        try:
            branch.repository.store_revision_signature(gpg.LoopbackGPGStrategy(None), b'FOO', b'A')
        except errors.NoSuchRevision:
            raise tests.TestNotApplicable('repository does not support signing non-presentrevisions')
    self.assertRaises(errors.NoSuchRevision, branch.repository.has_signature_for_revision_id, b'A')
    if wt.branch.repository._format.supports_setting_revision_ids:
        wt.commit('base', rev_id=b'A', allow_pointless=True)
        self.assertEqual(b'-----BEGIN PSEUDO-SIGNED CONTENT-----\nFOO-----END PSEUDO-SIGNED CONTENT-----\n', branch.repository.get_signature_text(b'A'))