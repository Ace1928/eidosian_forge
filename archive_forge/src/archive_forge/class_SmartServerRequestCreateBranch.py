import fastbencode as bencode
from ... import branch, errors, repository, urlutils
from ...controldir import network_format_registry
from .. import BzrProber
from ..bzrdir import BzrDir, BzrDirFormat
from .request import (FailedSmartServerResponse, SmartServerRequest,
class SmartServerRequestCreateBranch(SmartServerRequestBzrDir):

    def do(self, path, network_name):
        """Create a branch in the bzr dir at path.

        This operates precisely like 'bzrdir.create_branch'.

        If a bzrdir is not present, an exception is propogated
        rather than 'no branch' because these are different conditions (and
        this method should only be called after establishing that a bzr dir
        exists anyway).

        This is the initial version of this method introduced to the smart
        server for 1.13.

        :param path: The path to the bzrdir.
        :param network_name: The network name of the branch type to create.
        :return: ('ok', branch_format, repo_path, rich_root, tree_ref,
            external_lookup, repo_format)
        """
        bzrdir = BzrDir.open_from_transport(self.transport_from_client_path(path))
        format = branch.network_format_registry.get(network_name)
        bzrdir.branch_format = format
        result = format.initialize(bzrdir, name='')
        rich_root, tree_ref, external_lookup = self._format_to_capabilities(result.repository._format)
        branch_format = result._format.network_name()
        repo_format = result.repository._format.network_name()
        repo_path = self._repo_relpath(bzrdir.root_transport, result.repository)
        return SuccessfulSmartServerResponse((b'ok', branch_format, repo_path, rich_root, tree_ref, external_lookup, repo_format))