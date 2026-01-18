from breezy import branch, errors, uncommit
from breezy.tests import per_branch
def test_post_uncommit_not_to_origin(self):
    tree = self.make_branch_and_memory_tree('branch')
    tree.lock_write()
    tree.add('')
    revid = tree.commit('first revision')
    revid2 = tree.commit('second revision')
    revid3 = tree.commit('third revision')
    tree.unlock()
    branch.Branch.hooks.install_named_hook('post_uncommit', self.capture_post_uncommit_hook, None)
    uncommit.uncommit(tree.branch, revno=2)
    self.assertEqual([('post_uncommit', None, tree.branch.base, 3, revid3, 1, revid, None, True)], self.hook_calls)