import _thread
import errno
import io
import os
import sys
import time
import breezy
from ...lazy_import import lazy_import
import select
import socket
import weakref
from breezy import (
from breezy.i18n import gettext
from breezy.bzr.smart import client, protocol, request, signals, vfs
from breezy.transport import ssh
from ... import errors, osutils
class SmartClientSocketMedium(SmartClientStreamMedium):
    """A client medium using a socket.

    This class isn't usable directly.  Use one of its subclasses instead.
    """

    def __init__(self, base):
        SmartClientStreamMedium.__init__(self, base)
        self._socket = None
        self._connected = False

    def _accept_bytes(self, bytes):
        """See SmartClientMedium.accept_bytes."""
        self._ensure_connection()
        osutils.send_all(self._socket, bytes, self._report_activity)

    def _ensure_connection(self):
        """Connect this medium if not already connected."""
        raise NotImplementedError(self._ensure_connection)

    def _flush(self):
        """See SmartClientStreamMedium._flush().

        For sockets we do no flushing. For TCP sockets we may want to turn off
        TCP_NODELAY and add a means to do a flush, but that can be done in the
        future.
        """

    def _read_bytes(self, count):
        """See SmartClientMedium.read_bytes."""
        if not self._connected:
            raise errors.MediumNotConnected(self)
        return osutils.read_bytes_from_socket(self._socket, self._report_activity)

    def disconnect(self):
        """See SmartClientMedium.disconnect()."""
        if not self._connected:
            return
        self._socket.close()
        self._socket = None
        self._connected = False