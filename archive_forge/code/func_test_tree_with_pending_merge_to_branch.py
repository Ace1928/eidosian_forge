from breezy import branch as _mod_branch
from breezy import (controldir, errors, reconfigure, repository, tests,
from breezy.bzr import branch as _mod_bzrbranch
from breezy.bzr import vf_repository
def test_tree_with_pending_merge_to_branch(self):
    tree = self.make_branch_and_tree('tree')
    tree.commit('unchanged')
    other_tree = tree.controldir.sprout('other').open_workingtree()
    other_tree.commit('mergeable commit')
    tree.merge_from_branch(other_tree.branch)
    reconfiguration = reconfigure.Reconfigure.to_branch(tree.controldir)
    self.assertRaises(errors.UncommittedChanges, reconfiguration.apply)
    reconfiguration.apply(force=True)
    self.assertRaises(errors.NoWorkingTree, workingtree.WorkingTree.open, 'tree')