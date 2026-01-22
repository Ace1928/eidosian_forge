import logging
import weakref
from _dbus_bindings import (
from dbus.connection import Connection
from dbus.exceptions import DBusException
from dbus.lowlevel import HANDLER_RESULT_NOT_YET_HANDLED
from dbus._compat import is_py2
def _clean_up_signal_match(self, match):
    self.remove_match_string_non_blocking(str(match))
    watch = self._signal_sender_matches.pop(match, None)
    if watch is not None:
        watch.cancel()