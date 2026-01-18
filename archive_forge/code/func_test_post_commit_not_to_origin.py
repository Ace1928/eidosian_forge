from breezy import branch, delta, errors, revision, transport
from breezy.tests import per_branch
def test_post_commit_not_to_origin(self):
    tree = self.make_branch_and_memory_tree('branch')
    with tree.lock_write():
        tree.add('')
        revid = tree.commit('first revision')
        branch.Branch.hooks.install_named_hook('post_commit', self.capture_post_commit_hook, None)
        revid2 = tree.commit('second revision')
        self.assertEqual([('post_commit', None, tree.branch.base, 1, revid, 2, revid2, None, True)], self.hook_calls)