import fastbencode as bencode
from ... import errors
from ... import revision as _mod_revision
from ... import transport as _mod_transport
from ...controldir import ControlDir
from .request import (FailedSmartServerResponse, SmartServerRequest,
class SmartServerBranchRequestGetStackedOnURL(SmartServerBranchRequest):

    def do_with_branch(self, branch):
        stacked_on_url = branch.get_stacked_on_url()
        return SuccessfulSmartServerResponse((b'ok', stacked_on_url.encode('ascii')))