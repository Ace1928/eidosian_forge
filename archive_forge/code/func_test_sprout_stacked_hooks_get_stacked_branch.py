import os
from breezy import branch as _mod_branch
from breezy import errors, osutils
from breezy import revision as _mod_revision
from breezy import tests, urlutils
from breezy.bzr import branch as _mod_bzrbranch
from breezy.bzr import remote
from breezy.tests import features
from breezy.tests.per_branch import TestCaseWithBranch
def test_sprout_stacked_hooks_get_stacked_branch(self):
    tree = self.make_branch_and_tree('source')
    tree.commit('a commit')
    revid = tree.commit('a second commit')
    source = tree.branch
    target_transport = self.get_transport('target')
    self.hook_calls = []
    _mod_branch.Branch.hooks.install_named_hook('pre_change_branch_tip', self.assertBranchHookBranchIsStacked, None)
    try:
        dir = source.controldir.sprout(target_transport.base, source.last_revision(), possible_transports=[target_transport], source_branch=source, stacked=True)
    except _mod_branch.UnstackableBranchFormat:
        if not self.branch_format.supports_stacking():
            raise tests.TestNotApplicable("Format doesn't auto stack successfully.")
        else:
            raise
    result = dir.open_branch()
    self.assertEqual(revid, result.last_revision())
    self.assertEqual(source.base, result.get_stacked_on_url())
    if isinstance(result, remote.RemoteBranch):
        expected_calls = 2
    else:
        expected_calls = 1
    self.assertEqual(expected_calls, len(self.hook_calls))