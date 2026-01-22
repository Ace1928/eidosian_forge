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
def _read_bytes(self, count):
    """See SmartClientMedium.read_bytes."""
    if not self._connected:
        raise errors.MediumNotConnected(self)
    return osutils.read_bytes_from_socket(self._socket, self._report_activity)