from breezy import errors
from breezy.branch import BindingUnsupported, Branch
from breezy.controldir import ControlDir
from breezy.revision import NULL_REVISION
from breezy.tests import TestNotApplicable
from breezy.tests.per_interbranch import TestCaseWithInterBranch
def test_post_pull_nonempty_history(self):
    target = self.make_to_branch_and_memory_tree('target')
    target.lock_write()
    target.add('')
    rev1 = target.commit('rev 1')
    target.unlock()
    sourcedir = target.controldir.clone(self.get_url('source'))
    source = sourcedir.open_branch().create_memorytree()
    rev2 = source.commit('rev 2')
    Branch.hooks.install_named_hook('post_pull', self.capture_post_pull_hook, None)
    target.branch.pull(source.branch)
    self.assertEqual([('post_pull', source.branch, None, target.branch.base, 1, rev1, 2, rev2, True, None, True)], self.hook_calls)