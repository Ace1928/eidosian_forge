from breezy import gpg, tests
def assertSigned(self, repo, revision_id):
    """Assert that revision_id is signed in repo."""
    self.assertTrue(repo.has_signature_for_revision_id(revision_id))