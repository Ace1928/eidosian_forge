import logging
import weakref
from _dbus_bindings import (
from dbus.connection import Connection
from dbus.exceptions import DBusException
from dbus.lowlevel import HANDLER_RESULT_NOT_YET_HANDLED
from dbus._compat import is_py2
def activate_name_owner(self, bus_name):
    if bus_name is not None and bus_name[:1] != ':' and (bus_name != BUS_DAEMON_NAME):
        try:
            return self.get_name_owner(bus_name)
        except DBusException as e:
            if e.get_dbus_name() != _NAME_HAS_NO_OWNER:
                raise
            self.start_service_by_name(bus_name)
            return self.get_name_owner(bus_name)
    else:
        return bus_name