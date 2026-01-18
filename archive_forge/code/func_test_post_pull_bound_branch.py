from breezy import errors
from breezy.branch import BindingUnsupported, Branch
from breezy.controldir import ControlDir
from breezy.revision import NULL_REVISION
from breezy.tests import TestNotApplicable
from breezy.tests.per_interbranch import TestCaseWithInterBranch
def test_post_pull_bound_branch(self):
    target = self.make_to_branch('target')
    local = self.make_from_branch('local')
    try:
        local.bind(target)
    except BindingUnsupported:
        local = ControlDir.create_branch_convenience('local2')
        local.bind(target)
    source = self.make_from_branch('source')
    Branch.hooks.install_named_hook('post_pull', self.capture_post_pull_hook, None)
    local.pull(source)
    self.assertEqual([('post_pull', source, local.base, target.base, 0, NULL_REVISION, 0, NULL_REVISION, True, True, True)], self.hook_calls)