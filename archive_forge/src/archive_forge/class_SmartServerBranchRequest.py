import fastbencode as bencode
from ... import errors
from ... import revision as _mod_revision
from ... import transport as _mod_transport
from ...controldir import ControlDir
from .request import (FailedSmartServerResponse, SmartServerRequest,
class SmartServerBranchRequest(SmartServerRequest):
    """Base class for handling common branch request logic.
    """

    def do(self, path, *args):
        """Execute a request for a branch at path.

        All Branch requests take a path to the branch as their first argument.

        If the branch is a branch reference, NotBranchError is raised.

        :param path: The path for the repository as received from the
            client.
        :return: A SmartServerResponse from self.do_with_branch().
        """
        transport = self.transport_from_client_path(path)
        controldir = ControlDir.open_from_transport(transport)
        if controldir.get_branch_reference() is not None:
            raise errors.NotBranchError(transport.base)
        branch = controldir.open_branch(ignore_fallbacks=True)
        return self.do_with_branch(branch, *args)