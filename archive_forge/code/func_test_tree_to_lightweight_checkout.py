from breezy import branch as _mod_branch
from breezy import (controldir, errors, reconfigure, repository, tests,
from breezy.bzr import branch as _mod_bzrbranch
from breezy.bzr import vf_repository
def test_tree_to_lightweight_checkout(self):
    parent = self.make_branch('parent')
    tree = self.make_branch_and_tree('tree')
    reconfiguration = reconfigure.Reconfigure.to_lightweight_checkout(tree.controldir)
    self.assertRaises(reconfigure.NoBindLocation, reconfiguration.apply)
    tree.branch.set_parent(parent.base)
    reconfiguration = reconfigure.Reconfigure.to_lightweight_checkout(tree.controldir)
    reconfiguration.apply()
    tree2 = self.make_branch_and_tree('tree2')
    reconfiguration = reconfigure.Reconfigure.to_lightweight_checkout(tree2.controldir, parent.base)
    reconfiguration.apply()