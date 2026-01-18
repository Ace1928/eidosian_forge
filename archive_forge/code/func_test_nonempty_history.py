from breezy import branch as _mod_branch
from breezy import errors, revision, tests
from breezy.bzr import remote
from breezy.tests import test_server
def test_nonempty_history(self):
    branch = self.make_branch_with_revision_ids(b'one-\xc2\xb5', b'two-\xc2\xb5')
    hook_calls = self.install_logging_hook('post')
    branch.set_last_revision_info(1, b'one-\xc2\xb5')
    expected_params = _mod_branch.ChangeBranchTipParams(branch, 2, 1, b'two-\xc2\xb5', b'one-\xc2\xb5')
    self.assertHookCalls(expected_params, branch, hook_calls)