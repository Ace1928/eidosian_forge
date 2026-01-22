import logging
import weakref
from _dbus_bindings import (
from dbus.connection import Connection
from dbus.exceptions import DBusException
from dbus.lowlevel import HANDLER_RESULT_NOT_YET_HANDLED
from dbus._compat import is_py2
def add_match_string_non_blocking(self, rule):
    """Arrange for this application to receive messages on the bus that
        match the given rule. This version will not block, but any errors
        will be ignored.


        :Parameters:
            `rule` : str
                The match rule
        :Raises `DBusException`: on error.
        :Since: 0.80.0
        """
    self.call_async(BUS_DAEMON_NAME, BUS_DAEMON_PATH, BUS_DAEMON_IFACE, 'AddMatch', 's', (rule,), None, None)