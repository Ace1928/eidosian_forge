import fastbencode as bencode
from ... import errors
from ... import revision as _mod_revision
from ... import transport as _mod_transport
from ...controldir import ControlDir
from .request import (FailedSmartServerResponse, SmartServerRequest,
class SmartServerBranchGetParent(SmartServerBranchRequest):

    def do_with_branch(self, branch):
        """Return the parent of branch."""
        parent = branch._get_parent_location() or ''
        return SuccessfulSmartServerResponse((parent.encode('utf-8'),))