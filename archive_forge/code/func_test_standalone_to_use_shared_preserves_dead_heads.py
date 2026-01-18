from breezy import branch as _mod_branch
from breezy import (controldir, errors, reconfigure, repository, tests,
from breezy.bzr import branch as _mod_bzrbranch
from breezy.bzr import vf_repository
def test_standalone_to_use_shared_preserves_dead_heads(self):
    self.build_tree(['root/'])
    tree = self.make_branch_and_tree('root/tree')
    self.add_dead_head(tree)
    tree.commit('Hello', rev_id=b'hello-id')
    repo = self.make_repository('root', shared=True)
    reconfiguration = reconfigure.Reconfigure.to_use_shared(tree.controldir)
    reconfiguration.apply()
    tree = workingtree.WorkingTree.open('root/tree')
    message = repo.get_revision(b'dead-head-id').message
    self.assertEqual('Dead head', message)