from breezy import errors
from breezy.branch import BindingUnsupported, Branch
from breezy.controldir import ControlDir
from breezy.revision import NULL_REVISION
from breezy.tests import TestNotApplicable
from breezy.tests.per_interbranch import TestCaseWithInterBranch
def test_post_pull_empty_history(self):
    target = self.make_to_branch('target')
    source = self.make_from_branch('source')
    Branch.hooks.install_named_hook('post_pull', self.capture_post_pull_hook, None)
    target.pull(source)
    self.assertEqual([('post_pull', source, None, target.base, 0, NULL_REVISION, 0, NULL_REVISION, True, None, True)], self.hook_calls)