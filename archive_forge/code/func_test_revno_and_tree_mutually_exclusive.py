import os
from breezy import tests
from breezy.bzr.tests.matchers import ContainsNoVfsCalls
from breezy.errors import NoSuchRevision
def test_revno_and_tree_mutually_exclusive(self):
    wt = self.make_branch_and_tree('.')
    out, err = self.run_bzr('revno -r-2 --tree .', retcode=3)
    self.assertEqual('', out)
    self.assertEqual('brz: ERROR: --tree and --revision can not be used together\n', err)