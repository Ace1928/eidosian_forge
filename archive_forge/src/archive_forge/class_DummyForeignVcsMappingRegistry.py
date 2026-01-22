from .. import (branch, controldir, errors, foreign, lockable_files, lockdir,
from .. import transport as _mod_transport
from ..bzr import branch as bzrbranch
from ..bzr import bzrdir, groupcompress_repo, vf_repository
from ..bzr.pack_repo import PackCommitBuilder
class DummyForeignVcsMappingRegistry(foreign.VcsMappingRegistry):

    def revision_id_bzr_to_foreign(self, revid):
        if not revid.startswith(b'dummy-'):
            raise errors.InvalidRevisionId(revid, None)
        mapping_version = revid[len(b'dummy-'):len(b'dummy-vx')]
        mapping = self.get(mapping_version)
        return mapping.revision_id_bzr_to_foreign(revid)