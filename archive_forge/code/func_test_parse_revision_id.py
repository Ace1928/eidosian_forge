from .. import (branch, controldir, errors, foreign, lockable_files, lockdir,
from .. import transport as _mod_transport
from ..bzr import branch as bzrbranch
from ..bzr import bzrdir, groupcompress_repo, vf_repository
from ..bzr.pack_repo import PackCommitBuilder
def test_parse_revision_id(self):
    reg = foreign.ForeignVcsRegistry()
    vcs = DummyForeignVcs()
    reg.register('dummy', vcs, 'Dummy VCS')
    self.assertEqual(((b'some', b'foreign', b'revid'), DummyForeignVcsMapping(vcs)), reg.parse_revision_id(b'dummy-v1:some-foreign-revid'))