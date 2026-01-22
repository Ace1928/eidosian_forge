import fastbencode as bencode
from ... import branch, errors, repository, urlutils
from ...controldir import network_format_registry
from .. import BzrProber
from ..bzrdir import BzrDir, BzrDirFormat
from .request import (FailedSmartServerResponse, SmartServerRequest,
class SmartServerRequestBzrDir(SmartServerRequest):

    def do(self, path, *args):
        """Open a BzrDir at path, and return `self.do_bzrdir_request(*args)`."""
        try:
            self._bzrdir = BzrDir.open_from_transport(self.transport_from_client_path(path))
        except errors.NotBranchError as e:
            return FailedSmartServerResponse((b'nobranch',))
        return self.do_bzrdir_request(*args)

    def _boolean_to_yes_no(self, a_boolean):
        if a_boolean:
            return b'yes'
        else:
            return b'no'

    def _format_to_capabilities(self, repo_format):
        rich_root = self._boolean_to_yes_no(repo_format.rich_root_data)
        tree_ref = self._boolean_to_yes_no(repo_format.supports_tree_reference)
        external_lookup = self._boolean_to_yes_no(repo_format.supports_external_lookups)
        return (rich_root, tree_ref, external_lookup)

    def _repo_relpath(self, current_transport, repository):
        """Get the relative path for repository from current_transport."""
        relpath = repository.user_transport.relpath(current_transport.base)
        if len(relpath):
            segments = ['..'] * len(relpath.split('/'))
        else:
            segments = []
        return '/'.join(segments)