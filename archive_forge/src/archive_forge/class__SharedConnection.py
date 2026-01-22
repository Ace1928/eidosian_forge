import errno
import sys
from io import BytesIO
from stat import S_ISDIR
from typing import Any, Callable, Dict, TypeVar
from .. import errors, hooks, osutils, registry, ui, urlutils
from ..trace import mutter
class _SharedConnection:
    """A connection shared between several transports."""

    def __init__(self, connection=None, credentials=None, base=None):
        """Constructor.

        :param connection: An opaque object specific to each transport.

        :param credentials: An opaque object containing the credentials used to
            create the connection.
        """
        self.connection = connection
        self.credentials = credentials
        self.base = base