import os
from breezy import tests
from breezy.bzr.tests.matchers import ContainsNoVfsCalls
from breezy.errors import NoSuchRevision
def test_revno_tree_no_tree(self):
    b = self.make_branch('branch')
    out, err = self.run_bzr('revno --tree branch', retcode=3)
    self.assertEqual('', out)
    self.assertEqual('brz: ERROR: No WorkingTree exists for "branch".\n', err)