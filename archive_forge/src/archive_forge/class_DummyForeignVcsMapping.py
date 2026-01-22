from .. import (branch, controldir, errors, foreign, lockable_files, lockdir,
from .. import transport as _mod_transport
from ..bzr import branch as bzrbranch
from ..bzr import bzrdir, groupcompress_repo, vf_repository
from ..bzr.pack_repo import PackCommitBuilder
class DummyForeignVcsMapping(foreign.VcsMapping):
    """A simple mapping for the dummy Foreign VCS, for use with testing."""

    def __eq__(self, other):
        return isinstance(self, type(other))

    def revision_id_bzr_to_foreign(self, bzr_revid):
        return (tuple(bzr_revid[len(b'dummy-v1:'):].split(b'-')), self)

    def revision_id_foreign_to_bzr(self, foreign_revid):
        return b'dummy-v1:%s-%s-%s' % foreign_revid