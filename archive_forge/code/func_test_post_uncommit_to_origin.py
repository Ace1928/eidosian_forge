from breezy import branch, errors, uncommit
from breezy.tests import per_branch
def test_post_uncommit_to_origin(self):
    tree = self.make_branch_and_memory_tree('branch')
    tree.lock_write()
    tree.add('')
    revid = tree.commit('a revision')
    tree.unlock()
    branch.Branch.hooks.install_named_hook('post_uncommit', self.capture_post_uncommit_hook, None)
    uncommit.uncommit(tree.branch)
    self.assertEqual([('post_uncommit', None, tree.branch.base, 1, revid, 0, None, None, True)], self.hook_calls)