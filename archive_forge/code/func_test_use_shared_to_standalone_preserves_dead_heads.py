from breezy import branch as _mod_branch
from breezy import (controldir, errors, reconfigure, repository, tests,
from breezy.bzr import branch as _mod_bzrbranch
from breezy.bzr import vf_repository
def test_use_shared_to_standalone_preserves_dead_heads(self):
    tree = self.make_repository_tree()
    self.add_dead_head(tree)
    tree.commit('Hello', rev_id=b'hello-id')
    reconfigure.Reconfigure.to_standalone(tree.controldir).apply()
    tree = workingtree.WorkingTree.open('root/tree')
    repo = tree.branch.repository
    self.assertRaises(errors.NoSuchRevision, repo.get_revision, b'dead-head-id')