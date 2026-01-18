from breezy import branch as _mod_branch
from breezy import (controldir, errors, reconfigure, repository, tests,
from breezy.bzr import branch as _mod_bzrbranch
from breezy.bzr import vf_repository
def test_repo_to_tree(self):
    repo = self.make_repository('repo')
    reconfiguration = reconfigure.Reconfigure.to_tree(repo.controldir)
    reconfiguration.apply()
    workingtree.WorkingTree.open('repo')