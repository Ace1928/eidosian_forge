import fastbencode as bencode
from ... import errors
from ... import revision as _mod_revision
from ... import transport as _mod_transport
from ...controldir import ControlDir
from .request import (FailedSmartServerResponse, SmartServerRequest,
class SmartServerBranchRequestGetAllReferenceInfo(SmartServerBranchRequest):
    """Get the reference information.

    New in 3.1.
    """

    def do_with_branch(self, branch):
        all_reference_info = branch._get_all_reference_info()
        content = bencode.bencode([(key, value[0].encode('utf-8'), value[1].encode('utf-8') if value[1] else b'') for key, value in all_reference_info.items()])
        return SuccessfulSmartServerResponse((b'ok',), content)