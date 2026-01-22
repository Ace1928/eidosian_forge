import errno
import os.path
import socket
import sys
import threading
import time
from ... import errors, trace
from ... import transport as _mod_transport
from ...hooks import Hooks
from ...i18n import gettext
from ...lazy_import import lazy_import
from breezy.bzr.smart import (
from breezy.transport import (
from breezy import (
def _poll_active_connections(self, timeout=0.0):
    """Check to see if any active connections have finished.

        This will iterate through self._active_connections, and update any
        connections that are finished.

        :param timeout: The timeout to pass to thread.join(). By default, we
            set it to 0, so that we don't hang if threads are not done yet.
        :return: None
        """
    still_active = []
    for handler, thread in self._active_connections:
        thread.join(timeout)
        if thread.is_alive():
            still_active.append((handler, thread))
    self._active_connections = still_active