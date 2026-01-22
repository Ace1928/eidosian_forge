import fastbencode as bencode
from ... import branch, errors, repository, urlutils
from ...controldir import network_format_registry
from .. import BzrProber
from ..bzrdir import BzrDir, BzrDirFormat
from .request import (FailedSmartServerResponse, SmartServerRequest,
class SmartServerBzrDirRequestCheckoutMetaDir(SmartServerRequestBzrDir):
    """Get the format to use for checkouts.

    New in 2.5.

    :return: on success, a 3-tuple of network names for (control,
        repository, branch) directories, where '' signifies "not present".
        If this BzrDir contains a branch reference then this will fail with
        BranchReference; clients should resolve branch references before
        calling this RPC (they should not try to create a checkout of a
        checkout).
    """

    def do_bzrdir_request(self):
        try:
            branch_ref = self._bzrdir.get_branch_reference()
        except errors.NotBranchError:
            branch_ref = None
        if branch_ref is not None:
            return FailedSmartServerResponse((b'BranchReference',))
        control_format = self._bzrdir.checkout_metadir()
        control_name = control_format.network_name()
        if not control_format.fixed_components:
            branch_name = control_format.get_branch_format().network_name()
            repo_name = control_format.repository_format.network_name()
        else:
            branch_name = b''
            repo_name = b''
        return SuccessfulSmartServerResponse((control_name, repo_name, branch_name))