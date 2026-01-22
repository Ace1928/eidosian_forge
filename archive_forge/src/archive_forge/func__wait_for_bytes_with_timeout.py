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
def _wait_for_bytes_with_timeout(self, timeout_seconds):
    """Wait for more bytes to be read, but timeout if none available.

        This allows us to detect idle connections, and stop trying to read from
        them, without setting the socket itself to non-blocking. This also
        allows us to specify when we watch for idle timeouts.

        :return: None, this will raise ConnectionTimeout if we time out before
            data is available.
        """
    if getattr(self._in, 'fileno', None) is None or sys.platform == 'win32':
        return
    try:
        return self._wait_on_descriptor(self._in, timeout_seconds)
    except io.UnsupportedOperation:
        return