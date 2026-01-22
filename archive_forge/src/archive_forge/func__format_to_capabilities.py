import fastbencode as bencode
from ... import branch, errors, repository, urlutils
from ...controldir import network_format_registry
from .. import BzrProber
from ..bzrdir import BzrDir, BzrDirFormat
from .request import (FailedSmartServerResponse, SmartServerRequest,
def _format_to_capabilities(self, repo_format):
    rich_root = self._boolean_to_yes_no(repo_format.rich_root_data)
    tree_ref = self._boolean_to_yes_no(repo_format.supports_tree_reference)
    external_lookup = self._boolean_to_yes_no(repo_format.supports_external_lookups)
    return (rich_root, tree_ref, external_lookup)