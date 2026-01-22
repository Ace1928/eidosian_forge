import fastbencode as bencode
from ... import errors
from ... import revision as _mod_revision
from ... import transport as _mod_transport
from ...controldir import ControlDir
from .request import (FailedSmartServerResponse, SmartServerRequest,
class SmartServerBranchHeadsToFetch(SmartServerBranchRequest):

    def do_with_branch(self, branch):
        """Return the heads-to-fetch for a Branch as two bencoded lists.

        See Branch.heads_to_fetch.

        New in 2.4.
        """
        must_fetch, if_present_fetch = branch.heads_to_fetch()
        return SuccessfulSmartServerResponse((list(must_fetch), list(if_present_fetch)))