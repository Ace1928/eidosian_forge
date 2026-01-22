from .. import (branch, controldir, errors, foreign, lockable_files, lockdir,
from .. import transport as _mod_transport
from ..bzr import branch as bzrbranch
from ..bzr import bzrdir, groupcompress_repo, vf_repository
from ..bzr.pack_repo import PackCommitBuilder
class ForeignRevisionTests(tests.TestCase):
    """Tests for the ForeignRevision class."""

    def test_create(self):
        mapp = DummyForeignVcsMapping(DummyForeignVcs())
        rev = foreign.ForeignRevision((b'a', b'foreign', b'revid'), mapp, b'roundtripped-revid')
        self.assertEqual(b'', rev.inventory_sha1)
        self.assertEqual((b'a', b'foreign', b'revid'), rev.foreign_revid)
        self.assertEqual(mapp, rev.mapping)