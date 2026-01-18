from .. import (branch, controldir, errors, foreign, lockable_files, lockdir,
from .. import transport as _mod_transport
from ..bzr import branch as bzrbranch
from ..bzr import bzrdir, groupcompress_repo, vf_repository
from ..bzr.pack_repo import PackCommitBuilder
def test_parse_revision_id_unknown_mapping(self):
    reg = foreign.ForeignVcsRegistry()
    self.assertRaises(errors.InvalidRevisionId, reg.parse_revision_id, b'unknown-foreignrevid')