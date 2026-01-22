import errno
import sys
from io import BytesIO
from stat import S_ISDIR
from typing import Any, Callable, Dict, TypeVar
from .. import errors, hooks, osutils, registry, ui, urlutils
from ..trace import mutter
def _set_connection(self, connection, credentials=None):
    """Record a newly created connection with its associated credentials.

        Note: To ensure that connection is still shared after a temporary
        failure and a new one needs to be created, daughter classes should
        always call this method to set the connection and do so each time a new
        connection is created.

        Args:
          connection: An opaque object representing the connection used by
            the daughter class.
          credentials: An opaque object representing the credentials
            needed to create the connection.
        """
    self._shared_connection.connection = connection
    self._shared_connection.credentials = credentials
    for hook in self.hooks['post_connect']:
        hook(self)