from breezy import gpg, tests
def assertUnsigned(self, repo, revision_id):
    """Assert that revision_id is not signed in repo."""
    self.assertFalse(repo.has_signature_for_revision_id(revision_id))