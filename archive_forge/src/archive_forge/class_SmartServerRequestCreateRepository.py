import fastbencode as bencode
from ... import branch, errors, repository, urlutils
from ...controldir import network_format_registry
from .. import BzrProber
from ..bzrdir import BzrDir, BzrDirFormat
from .request import (FailedSmartServerResponse, SmartServerRequest,
class SmartServerRequestCreateRepository(SmartServerRequestBzrDir):

    def do(self, path, network_name, shared):
        """Create a repository in the bzr dir at path.

        This operates precisely like 'bzrdir.create_repository'.

        If a bzrdir is not present, an exception is propagated
        rather than 'no branch' because these are different conditions (and
        this method should only be called after establishing that a bzr dir
        exists anyway).

        This is the initial version of this method introduced to the smart
        server for 1.13.

        :param path: The path to the bzrdir.
        :param network_name: The network name of the repository type to create.
        :param shared: The value to pass create_repository for the shared
            parameter.
        :return: (ok, rich_root, tree_ref, external_lookup, network_name)
        """
        bzrdir = BzrDir.open_from_transport(self.transport_from_client_path(path))
        shared = shared == b'True'
        format = repository.network_format_registry.get(network_name)
        bzrdir.repository_format = format
        result = format.initialize(bzrdir, shared=shared)
        rich_root, tree_ref, external_lookup = self._format_to_capabilities(result._format)
        return SuccessfulSmartServerResponse((b'ok', rich_root, tree_ref, external_lookup, result._format.network_name()))