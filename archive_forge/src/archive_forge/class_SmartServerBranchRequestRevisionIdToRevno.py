import fastbencode as bencode
from ... import errors
from ... import revision as _mod_revision
from ... import transport as _mod_transport
from ...controldir import ControlDir
from .request import (FailedSmartServerResponse, SmartServerRequest,
class SmartServerBranchRequestRevisionIdToRevno(SmartServerBranchRequest):

    def do_with_branch(self, branch, revid):
        """Return branch.revision_id_to_revno().

        New in 2.5.

        The revno is encoded in decimal, the revision_id is encoded as utf8.
        """
        try:
            dotted_revno = branch.revision_id_to_dotted_revno(revid)
        except errors.NoSuchRevision:
            return FailedSmartServerResponse((b'NoSuchRevision', revid))
        except errors.GhostRevisionsHaveNoRevno as e:
            return FailedSmartServerResponse((b'GhostRevisionsHaveNoRevno', e.revision_id, e.ghost_revision_id))
        return SuccessfulSmartServerResponse((b'ok',) + tuple([b'%d' % x for x in dotted_revno]))