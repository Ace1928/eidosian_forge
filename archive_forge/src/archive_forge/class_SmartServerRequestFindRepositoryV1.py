import fastbencode as bencode
from ... import branch, errors, repository, urlutils
from ...controldir import network_format_registry
from .. import BzrProber
from ..bzrdir import BzrDir, BzrDirFormat
from .request import (FailedSmartServerResponse, SmartServerRequest,
class SmartServerRequestFindRepositoryV1(SmartServerRequestFindRepository):

    def do(self, path):
        """try to find a repository from path upwards

        This operates precisely like 'bzrdir.find_repository'.

        If a bzrdir is not present, an exception is propagated
        rather than 'no branch' because these are different conditions.

        This is the initial version of this method introduced with the smart
        server. Modern clients will try the V2 method that adds support for the
        supports_external_lookups attribute.

        :return: norepository or ok, relpath.
        """
        try:
            path, rich_root, tree_ref, external_lookup, name = self._find(path)
            return SuccessfulSmartServerResponse((b'ok', path.encode('utf-8'), rich_root, tree_ref))
        except errors.NoRepositoryPresent:
            return FailedSmartServerResponse((b'norepository',))