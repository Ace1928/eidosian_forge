import fastbencode as bencode
from ... import errors
from ... import revision as _mod_revision
from ... import transport as _mod_transport
from ...controldir import ControlDir
from .request import (FailedSmartServerResponse, SmartServerRequest,
class SmartServerBranchRequestLastRevisionInfo(SmartServerBranchRequest):

    def do_with_branch(self, branch):
        """Return branch.last_revision_info().

        The revno is encoded in decimal, the revision_id is encoded as utf8.
        """
        revno, last_revision = branch.last_revision_info()
        return SuccessfulSmartServerResponse((b'ok', str(revno).encode('ascii'), last_revision))