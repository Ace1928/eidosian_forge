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
def test_sprout_from_hpss_preserves_format(self):
    """repo.sprout from a smart server preserves the repository format."""
    remote_repo = self.make_remote_repository('remote')
    local_bzrdir = self.make_controldir('local')
    try:
        local_repo = remote_repo.sprout(local_bzrdir)
    except errors.TransportNotPossible:
        raise tests.TestNotApplicable('Cannot lock_read old formats like AllInOne over HPSS.')
    remote_backing_repo = controldir.ControlDir.open(self.get_vfs_only_url('remote')).open_repository()
    self.assertEqual(remote_backing_repo._format.network_name(), local_repo._format.network_name())