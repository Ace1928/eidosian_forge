import logging
import weakref
from _dbus_bindings import (
from dbus.connection import Connection
from dbus.exceptions import DBusException
from dbus.lowlevel import HANDLER_RESULT_NOT_YET_HANDLED
from dbus._compat import is_py2
def add_signal_receiver(self, handler_function, signal_name=None, dbus_interface=None, bus_name=None, path=None, **keywords):
    named_service = keywords.pop('named_service', None)
    if named_service is not None:
        if bus_name is not None:
            raise TypeError('bus_name and named_service cannot both be specified')
        bus_name = named_service
        from warnings import warn
        warn('Passing the named_service parameter to add_signal_receiver by name is deprecated: please use positional parameters', DeprecationWarning, stacklevel=2)
    match = super(BusConnection, self).add_signal_receiver(handler_function, signal_name, dbus_interface, bus_name, path, **keywords)
    if bus_name is not None and bus_name != BUS_DAEMON_NAME:
        if bus_name[:1] == ':':

            def callback(new_owner):
                if new_owner == '':
                    match.remove()
        else:
            callback = match.set_sender_name_owner
        watch = self.watch_name_owner(bus_name, callback)
        self._signal_sender_matches[match] = watch
    self.add_match_string(str(match))
    return match