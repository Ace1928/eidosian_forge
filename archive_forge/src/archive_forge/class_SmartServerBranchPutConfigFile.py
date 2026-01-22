import fastbencode as bencode
from ... import errors
from ... import revision as _mod_revision
from ... import transport as _mod_transport
from ...controldir import ControlDir
from .request import (FailedSmartServerResponse, SmartServerRequest,
class SmartServerBranchPutConfigFile(SmartServerBranchRequest):
    """Set the configuration data for a branch.

    New in 2.5.
    """

    def do_with_branch(self, branch, branch_token, repo_token):
        """Set the content of branch.conf.

        The body is not utf8 decoded - its the literal bytestream for disk.
        """
        self._branch = branch
        self._branch_token = branch_token
        self._repo_token = repo_token
        return None

    def do_body(self, body_bytes):
        with self._branch.repository.lock_write(token=self._repo_token), self._branch.lock_write(token=self._branch_token):
            self._branch.control_transport.put_bytes('branch.conf', body_bytes)
        return SuccessfulSmartServerResponse((b'ok',))