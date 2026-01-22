from breezy import branch as _mod_branch
from breezy import errors, revision, tests
from breezy.bzr import remote
from breezy.tests import test_server
class ChangeBranchTipTestCase(tests.TestCaseWithMemoryTransport):
    """Base TestCase for testing pre/post_change_branch_tip hooks."""

    def install_logging_hook(self, prefix):
        """Add a hook that logs calls made to it.

        :returns: the list that the calls will be appended to.
        """
        hook_calls = []
        _mod_branch.Branch.hooks.install_named_hook(prefix + '_change_branch_tip', hook_calls.append, None)
        return hook_calls

    def make_branch_with_revision_ids(self, *revision_ids):
        """Makes a branch with the given commits."""
        tree = self.make_branch_and_memory_tree('source')
        tree.lock_write()
        tree.add('')
        for revision_id in revision_ids:
            tree.commit('Message of ' + revision_id.decode('utf8'), rev_id=revision_id)
        tree.unlock()
        branch = tree.branch
        return branch

    def assertHookCalls(self, expected_params, branch, hook_calls=None, pre=False):
        if hook_calls is None:
            hook_calls = self.hook_calls
        if isinstance(branch, remote.RemoteBranch):
            if pre:
                offset = 0
            else:
                offset = 1
            self.assertEqual(expected_params, hook_calls[offset])
            self.assertEqual(2, len(hook_calls))
        else:
            self.assertEqual([expected_params], hook_calls)