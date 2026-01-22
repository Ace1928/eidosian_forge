from collections.abc import Mapping
import copy
import logging
import sys
import traceback
from oslo_serialization import jsonutils
from oslo_utils import timeutils
import oslo_messaging
from oslo_messaging import _utils as utils
class ConnectionContext(Connection):
    """The class that is actually returned to the create_connection() caller.

    This is essentially a wrapper around Connection that supports 'with'.
    It can also return a new Connection, or one from a pool.

    The function will also catch when an instance of this class is to be
    deleted.  With that we can return Connections to the pool on exceptions
    and so forth without making the caller be responsible for catching them.
    If possible the function makes sure to return a connection to the pool.
    """

    def __init__(self, connection_pool, purpose, retry):
        """Create a new connection, or get one from the pool."""
        self.connection = None
        self.connection_pool = connection_pool
        pooled = purpose == PURPOSE_SEND
        if pooled:
            self.connection = connection_pool.get(retry=retry)
        else:
            self.connection = connection_pool.create(purpose)
        self.pooled = pooled
        self.connection.pooled = pooled

    def __enter__(self):
        """When with ConnectionContext() is used, return self."""
        return self

    def _done(self):
        """If the connection came from a pool, clean it up and put it back.
        If it did not come from a pool, close it.
        """
        if self.connection:
            if self.pooled:
                try:
                    self.connection.reset()
                except Exception:
                    LOG.exception('Fail to reset the connection, drop it')
                    try:
                        self.connection.close()
                    except Exception as exc:
                        LOG.debug('pooled conn close failure (ignored): %s', str(exc))
                    self.connection = self.connection_pool.create()
                finally:
                    self.connection_pool.put(self.connection)
            else:
                try:
                    self.connection.close()
                except Exception as exc:
                    LOG.debug('pooled conn close failure (ignored): %s', str(exc))
            self.connection = None

    def __exit__(self, exc_type, exc_value, tb):
        """End of 'with' statement.  We're done here."""
        self._done()

    def __del__(self):
        """Caller is done with this connection.  Make sure we cleaned up."""
        self._done()

    def close(self):
        """Caller is done with this connection."""
        self._done()

    def __getattr__(self, key):
        """Proxy all other calls to the Connection instance."""
        if self.connection:
            return getattr(self.connection, key)
        else:
            raise InvalidRPCConnectionReuse()