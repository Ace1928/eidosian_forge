import os
from breezy import tests
def test_pack_accepts_branch_url(self):
    """pack command accepts the url to a branch."""
    self.make_branch('branch')
    out, err = self.run_bzr('pack branch')
    self.assertEqual('', out)
    self.assertEqual('', err)