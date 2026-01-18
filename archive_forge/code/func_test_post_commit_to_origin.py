from breezy import branch, delta, errors, revision, transport
from breezy.tests import per_branch
def test_post_commit_to_origin(self):
    tree = self.make_branch_and_memory_tree('branch')
    branch.Branch.hooks.install_named_hook('post_commit', self.capture_post_commit_hook, None)
    tree.lock_write()
    tree.add('')
    revid = tree.commit('a revision')
    self.assertEqual([('post_commit', None, tree.branch.base, 0, revision.NULL_REVISION, 1, revid, None, True)], self.hook_calls)
    tree.unlock()