from breezy import branch as _mod_branch
from breezy import (controldir, errors, reconfigure, repository, tests,
from breezy.bzr import branch as _mod_bzrbranch
from breezy.bzr import vf_repository
def test_shared_repo_to_lightweight_checkout(self):
    repo = self.make_repository('repo', shared=True)
    reconfiguration = reconfigure.Reconfigure.to_lightweight_checkout(repo.controldir)
    self.assertRaises(reconfigure.NoBindLocation, reconfiguration.apply)
    self.make_branch('branch')
    reconfiguration = reconfigure.Reconfigure.to_lightweight_checkout(repo.controldir, 'branch')
    reconfiguration.apply()
    workingtree.WorkingTree.open('repo')
    repository.Repository.open('repo')