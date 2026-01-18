import re
from io import BytesIO
from ... import branch as _mod_branch
from ... import commit, controldir
from ... import delta as _mod_delta
from ... import errors, gpg, info, repository
from ... import revision as _mod_revision
from ... import tests, transport, upgrade, workingtree
from ...bzr import branch as _mod_bzrbranch
from ...bzr import inventory, knitpack_repo, remote
from ...bzr import repository as bzrrepository
from .. import per_repository, test_server
from ..matchers import *
def test_sprout_branch_from_hpss_preserves_shared_repo_format(self):
    """branch.sprout from a smart server preserves the repository format of
        a branch from a shared repository.
        """
    if not self.repository_format.supports_leaving_lock:
        raise tests.TestNotApplicable('Format can not be used over HPSS')
    remote_repo = self.make_remote_repository('remote', shared=True)
    remote_backing_repo = controldir.ControlDir.open(self.get_vfs_only_url('remote')).open_repository()
    from breezy.bzr.fullhistory import BzrBranchFormat5
    format = remote_backing_repo.controldir.cloning_metadir()
    format._branch_format = BzrBranchFormat5()
    remote_transport = remote_repo.controldir.root_transport.clone('branch')
    controldir.ControlDir.create_branch_convenience(remote_transport.base, force_new_repo=False, format=format)
    remote_branch = controldir.ControlDir.open_from_transport(remote_transport).open_branch()
    try:
        local_bzrdir = remote_branch.controldir.sprout('local')
    except errors.TransportNotPossible:
        raise tests.TestNotApplicable('Cannot lock_read old formats like AllInOne over HPSS.')
    local_repo = local_bzrdir.open_repository()
    self.assertEqual(remote_backing_repo._format, local_repo._format)