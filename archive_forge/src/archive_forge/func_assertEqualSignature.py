from breezy import gpg, tests
from breezy.bzr.testament import Testament
from breezy.controldir import ControlDir
def assertEqualSignature(self, repo, revision_id):
    """Assert a signature is stored correctly in repository."""
    self.assertEqual(b'-----BEGIN PSEUDO-SIGNED CONTENT-----\n' + Testament.from_revision(repo, revision_id).as_short_text() + b'-----END PSEUDO-SIGNED CONTENT-----\n', repo.get_signature_text(revision_id))