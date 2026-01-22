from breezy import branch as _mod_branch
from breezy import errors, revision, tests
from breezy.bzr import remote
from breezy.tests import test_server
class TestPreChangeBranchTip(ChangeBranchTipTestCase):
    """Tests for pre_change_branch_tip hook.

    Most of these tests are very similar to the tests in
    TestPostChangeBranchTip.
    """

    def test_hook_runs_before_change(self):
        """The hook runs *before* the branch's last_revision_info has changed.
        """
        branch = self.make_branch_with_revision_ids(b'revid-one')

        def assertBranchAtRevision1(params):
            self.assertEqual((1, b'revid-one'), params.branch.last_revision_info())
        _mod_branch.Branch.hooks.install_named_hook('pre_change_branch_tip', assertBranchAtRevision1, None)
        branch.set_last_revision_info(0, revision.NULL_REVISION)

    def test_hook_failure_prevents_change(self):
        """If a hook raises an exception, the change does not take effect."""
        branch = self.make_branch_with_revision_ids(b'one-\xc2\xb5', b'two-\xc2\xb5')

        class PearShapedError(Exception):
            pass

        def hook_that_raises(params):
            raise PearShapedError()
        _mod_branch.Branch.hooks.install_named_hook('pre_change_branch_tip', hook_that_raises, None)
        hook_failed_exc = self.assertRaises(PearShapedError, branch.set_last_revision_info, 0, revision.NULL_REVISION)
        self.assertEqual((2, b'two-\xc2\xb5'), branch.last_revision_info())

    def test_empty_history(self):
        branch = self.make_branch('source')
        hook_calls = self.install_logging_hook('pre')
        branch.set_last_revision_info(0, revision.NULL_REVISION)
        expected_params = _mod_branch.ChangeBranchTipParams(branch, 0, 0, revision.NULL_REVISION, revision.NULL_REVISION)
        self.assertHookCalls(expected_params, branch, hook_calls, pre=True)

    def test_nonempty_history(self):
        branch = self.make_branch_with_revision_ids(b'one-\xc2\xb5', b'two-\xc2\xb5')
        hook_calls = self.install_logging_hook('pre')
        branch.set_last_revision_info(1, b'one-\xc2\xb5')
        expected_params = _mod_branch.ChangeBranchTipParams(branch, 2, 1, b'two-\xc2\xb5', b'one-\xc2\xb5')
        self.assertHookCalls(expected_params, branch, hook_calls, pre=True)

    def test_branch_is_locked(self):
        branch = self.make_branch('source')

        def assertBranchIsLocked(params):
            self.assertTrue(params.branch.is_locked())
        _mod_branch.Branch.hooks.install_named_hook('pre_change_branch_tip', assertBranchIsLocked, None)
        branch.set_last_revision_info(0, revision.NULL_REVISION)

    def test_calls_all_hooks_no_errors(self):
        """If multiple hooks are registered, all are called (if none raise
        errors).
        """
        branch = self.make_branch('source')
        hook_calls_1 = self.install_logging_hook('pre')
        hook_calls_2 = self.install_logging_hook('pre')
        self.assertIsNot(hook_calls_1, hook_calls_2)
        branch.set_last_revision_info(0, revision.NULL_REVISION)
        if isinstance(branch, remote.RemoteBranch):
            count = 2
        else:
            count = 1
        self.assertEqual(len(hook_calls_1), count)
        self.assertEqual(len(hook_calls_2), count)

    def test_explicit_reject_by_hook(self):
        """If a hook raises TipChangeRejected, the change does not take effect.

        TipChangeRejected exceptions are propagated, not wrapped in HookFailed.
        """
        branch = self.make_branch_with_revision_ids(b'one-\xc2\xb5', b'two-\xc2\xb5')

        def hook_that_rejects(params):
            raise errors.TipChangeRejected('rejection message')
        _mod_branch.Branch.hooks.install_named_hook('pre_change_branch_tip', hook_that_rejects, None)
        self.assertRaises(errors.TipChangeRejected, branch.set_last_revision_info, 0, revision.NULL_REVISION)
        self.assertEqual((2, b'two-\xc2\xb5'), branch.last_revision_info())