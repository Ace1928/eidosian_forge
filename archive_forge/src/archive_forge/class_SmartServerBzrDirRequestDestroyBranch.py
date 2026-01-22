import fastbencode as bencode
from ... import branch, errors, repository, urlutils
from ...controldir import network_format_registry
from .. import BzrProber
from ..bzrdir import BzrDir, BzrDirFormat
from .request import (FailedSmartServerResponse, SmartServerRequest,
class SmartServerBzrDirRequestDestroyBranch(SmartServerRequestBzrDir):

    def do_bzrdir_request(self, name=None):
        """Destroy the branch with the specified name.

        New in 2.5.0.
        :return: On success, 'ok'.
        """
        try:
            self._bzrdir.destroy_branch(name.decode('utf-8') if name is not None else None)
        except errors.NotBranchError as e:
            return FailedSmartServerResponse((b'nobranch',))
        return SuccessfulSmartServerResponse((b'ok',))