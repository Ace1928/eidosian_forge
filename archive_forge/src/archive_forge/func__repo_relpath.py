import fastbencode as bencode
from ... import branch, errors, repository, urlutils
from ...controldir import network_format_registry
from .. import BzrProber
from ..bzrdir import BzrDir, BzrDirFormat
from .request import (FailedSmartServerResponse, SmartServerRequest,
def _repo_relpath(self, current_transport, repository):
    """Get the relative path for repository from current_transport."""
    relpath = repository.user_transport.relpath(current_transport.base)
    if len(relpath):
        segments = ['..'] * len(relpath.split('/'))
    else:
        segments = []
    return '/'.join(segments)