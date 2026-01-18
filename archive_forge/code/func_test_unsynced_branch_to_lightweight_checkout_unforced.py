from breezy import branch as _mod_branch
from breezy import (controldir, errors, reconfigure, repository, tests,
from breezy.bzr import branch as _mod_bzrbranch
from breezy.bzr import vf_repository
def test_unsynced_branch_to_lightweight_checkout_unforced(self):
    reconfiguration = self.make_unsynced_branch_reconfiguration()
    self.assertRaises(reconfigure.UnsyncedBranches, reconfiguration.apply)