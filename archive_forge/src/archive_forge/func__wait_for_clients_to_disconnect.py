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
def _wait_for_clients_to_disconnect(self):
    self._poll_active_connections()
    if not self._active_connections:
        return
    trace.note(gettext('Waiting for %d client(s) to finish') % (len(self._active_connections),))
    t_next_log = self._timer() + self._LOG_WAITING_TIMEOUT
    while self._active_connections:
        now = self._timer()
        if now >= t_next_log:
            trace.note(gettext('Still waiting for %d client(s) to finish') % (len(self._active_connections),))
            t_next_log = now + self._LOG_WAITING_TIMEOUT
        self._poll_active_connections(self._SHUTDOWN_POLL_TIMEOUT)