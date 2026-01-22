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
class SmartClientAlreadyConnectedSocketMedium(SmartClientSocketMedium):
    """A client medium for an already connected socket.

    Note that this class will assume it "owns" the socket, so it will close it
    when its disconnect method is called.
    """

    def __init__(self, base, sock):
        SmartClientSocketMedium.__init__(self, base)
        self._socket = sock
        self._connected = True

    def _ensure_connection(self):
        pass