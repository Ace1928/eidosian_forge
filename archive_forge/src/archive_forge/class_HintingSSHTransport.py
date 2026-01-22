imported from breezy.bzr.smart.
from io import BytesIO
from .. import config, debug, errors, trace, transport, urlutils
from ..bzr import remote
from ..bzr.smart import client, medium
class HintingSSHTransport(transport.Transport):
    """Simple transport that handles ssh:// and points out bzr+ssh:// and git+ssh://."""

    def __init__(self, url):
        raise transport.UnsupportedProtocol(url, 'Use bzr+ssh for Bazaar operations over SSH, e.g. "bzr+%s". Use git+ssh for Git operations over SSH, e.g. "git+%s".' % (url, url))