import fastbencode as bencode
from ... import errors
from ... import revision as _mod_revision
from ... import transport as _mod_transport
from ...controldir import ControlDir
from .request import (FailedSmartServerResponse, SmartServerRequest,
class SmartServerBranchGetConfigFile(SmartServerBranchRequest):

    def do_with_branch(self, branch):
        """Return the content of branch.conf

        The body is not utf8 decoded - its the literal bytestream from disk.
        """
        try:
            content = branch.control_transport.get_bytes('branch.conf')
        except _mod_transport.NoSuchFile:
            content = b''
        return SuccessfulSmartServerResponse((b'ok',), content)