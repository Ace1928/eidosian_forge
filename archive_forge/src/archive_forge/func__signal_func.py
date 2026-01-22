import logging
import threading
import weakref
from _dbus_bindings import (
from dbus.exceptions import DBusException
from dbus.lowlevel import (
from dbus.proxies import ProxyObject
from dbus._compat import is_py2, is_py3
from _dbus_bindings import String
def _signal_func(self, message):
    """D-Bus filter function. Handle signals by dispatching to Python
        callbacks kept in the match-rule tree.
        """
    if not isinstance(message, SignalMessage):
        return HANDLER_RESULT_NOT_YET_HANDLED
    dbus_interface = message.get_interface()
    path = message.get_path()
    signal_name = message.get_member()
    for match in self._iter_easy_matches(path, dbus_interface, signal_name):
        match.maybe_handle_message(message)
    if dbus_interface == LOCAL_IFACE and path == LOCAL_PATH and (signal_name == 'Disconnected'):
        for cb in self.__call_on_disconnection:
            try:
                cb(self)
            except Exception:
                logging.basicConfig()
                _logger.error('Exception in handler for Disconnected signal:', exc_info=1)
    return HANDLER_RESULT_NOT_YET_HANDLED