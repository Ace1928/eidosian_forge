import fastbencode as bencode
from ... import errors
from ... import revision as _mod_revision
from ... import transport as _mod_transport
from ...controldir import ControlDir
from .request import (FailedSmartServerResponse, SmartServerRequest,
class SmartServerBranchSetTagsBytes(SmartServerLockedBranchRequest):

    def __init__(self, backing_transport, root_client_path='/', jail_root=None):
        SmartServerLockedBranchRequest.__init__(self, backing_transport, root_client_path, jail_root)
        self.locked = False

    def do_with_locked_branch(self, branch):
        """Call _set_tags_bytes for a branch.

        New in 1.18.
        """
        self.branch = branch
        self.branch.lock_write()
        self.locked = True

    def do_body(self, bytes):
        self.branch._set_tags_bytes(bytes)
        return SuccessfulSmartServerResponse(())

    def do_end(self):
        if not self.locked:
            return
        try:
            return SmartServerLockedBranchRequest.do_end(self)
        finally:
            self.branch.unlock()