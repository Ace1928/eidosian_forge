import logging
import threading
import weakref
from _dbus_bindings import (
from dbus.exceptions import DBusException
from dbus.lowlevel import (
from dbus.proxies import ProxyObject
from dbus._compat import is_py2, is_py3
from _dbus_bindings import String
def _iter_easy_matches(self, path, dbus_interface, member):
    if path is not None:
        path_keys = (None, path)
    else:
        path_keys = (None,)
    if dbus_interface is not None:
        interface_keys = (None, dbus_interface)
    else:
        interface_keys = (None,)
    if member is not None:
        member_keys = (None, member)
    else:
        member_keys = (None,)
    for path in path_keys:
        by_interface = self._signal_recipients_by_object_path.get(path)
        if by_interface is None:
            continue
        for dbus_interface in interface_keys:
            by_member = by_interface.get(dbus_interface, None)
            if by_member is None:
                continue
            for member in member_keys:
                matches = by_member.get(member, None)
                if matches is None:
                    continue
                for m in matches:
                    yield m