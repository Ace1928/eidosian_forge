from __future__ import absolute_import
from .packages.six.moves.http_client import IncompleteRead as httplib_IncompleteRead
class EmptyPoolError(PoolError):
    """Raised when a pool runs out of connections and no more are allowed."""
    pass