import logging
import _dbus_bindings
from dbus._expat_introspect_parser import process_introspection_data
from dbus.exceptions import (
from _dbus_bindings import (
from dbus._compat import is_py2
def _introspect_block(self):
    self._introspect_lock.acquire()
    try:
        if self._pending_introspect is not None:
            self._pending_introspect.block()
    finally:
        self._introspect_lock.release()