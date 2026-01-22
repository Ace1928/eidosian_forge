from .. import (branch, controldir, errors, foreign, lockable_files, lockdir,
from .. import transport as _mod_transport
from ..bzr import branch as bzrbranch
from ..bzr import bzrdir, groupcompress_repo, vf_repository
from ..bzr.pack_repo import PackCommitBuilder
class DummyForeignVcs(foreign.ForeignVcs):
    """A dummy Foreign VCS, for use with testing.

    It has revision ids that are a tuple with three strings.
    """

    def __init__(self):
        self.mapping_registry = DummyForeignVcsMappingRegistry()
        self.mapping_registry.register(b'v1', DummyForeignVcsMapping(self), 'Version 1')
        self.abbreviation = 'dummy'

    def show_foreign_revid(self, foreign_revid):
        return {'dummy ding': '%s/%s\\%s' % foreign_revid}

    def serialize_foreign_revid(self, foreign_revid):
        return '%s|%s|%s' % foreign_revid