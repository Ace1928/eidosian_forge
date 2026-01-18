from breezy import branch, delta, errors, revision, transport
from breezy.tests import per_branch
def test_pre_commit_passes(self):
    tree = self.make_branch_and_memory_tree('branch')
    with tree.lock_write():
        tree.add('')
        branch.Branch.hooks.install_named_hook('pre_commit', self.capture_pre_commit_hook, None)
        revid1 = tree.commit('first revision')
        revid2 = tree.commit('second revision')
        root_delta = self.get_rootfull_delta(tree.branch.repository, revid1)
        empty_delta = tree.branch.repository.get_revision_delta(revid2)
        self.assertEqual([('pre_commit', 0, revision.NULL_REVISION, 1, revid1, root_delta), ('pre_commit', 1, revid1, 2, revid2, empty_delta)], self.hook_calls)