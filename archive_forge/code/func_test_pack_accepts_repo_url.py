import os
from breezy import tests
def test_pack_accepts_repo_url(self):
    """pack command accepts the url to a branch."""
    self.make_repository('repository')
    out, err = self.run_bzr('pack repository')
    self.assertEqual('', out)
    self.assertEqual('', err)