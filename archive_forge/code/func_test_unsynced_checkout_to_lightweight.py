from breezy import branch as _mod_branch
from breezy import (controldir, errors, reconfigure, repository, tests,
from breezy.bzr import branch as _mod_bzrbranch
from breezy.bzr import vf_repository
def test_unsynced_checkout_to_lightweight(self):
    checkout, parent, reconfiguration = self.make_unsynced_checkout()
    self.assertRaises(reconfigure.UnsyncedBranches, reconfiguration.apply)