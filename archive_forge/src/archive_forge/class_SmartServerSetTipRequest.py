import fastbencode as bencode
from ... import errors
from ... import revision as _mod_revision
from ... import transport as _mod_transport
from ...controldir import ControlDir
from .request import (FailedSmartServerResponse, SmartServerRequest,
class SmartServerSetTipRequest(SmartServerLockedBranchRequest):
    """Base class for handling common branch request logic for requests that
    update the branch tip.
    """

    def do_with_locked_branch(self, branch, *args):
        try:
            return self.do_tip_change_with_locked_branch(branch, *args)
        except errors.TipChangeRejected as e:
            msg = e.msg
            if isinstance(msg, str):
                msg = msg.encode('utf-8')
            return FailedSmartServerResponse((b'TipChangeRejected', msg))