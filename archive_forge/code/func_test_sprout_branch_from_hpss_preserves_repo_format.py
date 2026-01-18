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
def test_sprout_branch_from_hpss_preserves_repo_format(self):
    """branch.sprout from a smart server preserves the repository format.
        """
    if not self.repository_format.supports_leaving_lock:
        raise tests.TestNotApplicable('Format can not be used over HPSS')
    remote_repo = self.make_remote_repository('remote')
    remote_branch = remote_repo.controldir.create_branch()
    try:
        local_bzrdir = remote_branch.controldir.sprout('local')
    except errors.TransportNotPossible:
        raise tests.TestNotApplicable('Cannot lock_read old formats like AllInOne over HPSS.')
    local_repo = local_bzrdir.open_repository()
    remote_backing_repo = controldir.ControlDir.open(self.get_vfs_only_url('remote')).open_repository()
    self.assertEqual(remote_backing_repo._format, local_repo._format)