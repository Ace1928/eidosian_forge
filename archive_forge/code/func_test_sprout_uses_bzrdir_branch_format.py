import os
from breezy import branch as _mod_branch
from breezy import errors, osutils
from breezy import revision as _mod_revision
from breezy import tests, urlutils
from breezy.bzr import branch as _mod_bzrbranch
from breezy.bzr import remote
from breezy.tests import features
from breezy.tests.per_branch import TestCaseWithBranch
def test_sprout_uses_bzrdir_branch_format(self):
    if isinstance(self.branch_format, _mod_bzrbranch.BranchReferenceFormat):
        raise tests.TestNotApplicable('cannot sprout to a reference')
    source = tests.TestCaseWithTransport.make_branch(self, 'old-branch', format='knit')
    target_bzrdir = self.make_controldir('target')
    target_bzrdir.create_repository()
    result_format = self.branch_format
    if isinstance(target_bzrdir, remote.RemoteBzrDir):
        target_bzrdir._format.set_branch_format(_mod_bzrbranch.BzrBranchFormat6())
        result_format = target_bzrdir._format.get_branch_format()
    target = source.sprout(target_bzrdir)
    if isinstance(target, remote.RemoteBranch):
        target._ensure_real()
        target = target._real_branch
    if isinstance(result_format, remote.RemoteBranchFormat):
        result_format = result_format._custom_format
    self.assertIs(result_format.__class__, target._format.__class__)