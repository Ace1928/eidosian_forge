from __future__ import absolute_import
from .packages.six.moves.http_client import IncompleteRead as httplib_IncompleteRead
class HostChangedError(RequestError):
    """Raised when an existing pool gets a request for a foreign host."""

    def __init__(self, pool, url, retries=3):
        message = 'Tried to open a foreign host with url: %s' % url
        RequestError.__init__(self, pool, url, message)
        self.retries = retries