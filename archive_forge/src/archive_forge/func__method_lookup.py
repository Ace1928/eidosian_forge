import sys
import logging
import threading
import traceback
import _dbus_bindings
from dbus import (
from dbus.decorators import method, signal
from dbus.exceptions import (
from dbus.lowlevel import ErrorMessage, MethodReturnMessage, MethodCallMessage
from dbus.proxies import LOCAL_PATH
from dbus._compat import is_py2
def _method_lookup(self, method_name, dbus_interface):
    """Walks the Python MRO of the given class to find the method to invoke.

    Returns two methods, the one to call, and the one it inherits from which
    defines its D-Bus interface name, signature, and attributes.
    """
    parent_method = None
    candidate_class = None
    successful = False
    if dbus_interface:
        for cls in self.__class__.__mro__:
            if not candidate_class and method_name in cls.__dict__:
                if '_dbus_is_method' in cls.__dict__[method_name].__dict__ and '_dbus_interface' in cls.__dict__[method_name].__dict__:
                    if cls.__dict__[method_name]._dbus_interface == dbus_interface:
                        candidate_class = cls
                        parent_method = cls.__dict__[method_name]
                        successful = True
                        break
                    else:
                        pass
                else:
                    candidate_class = cls
            if candidate_class and method_name in cls.__dict__ and ('_dbus_is_method' in cls.__dict__[method_name].__dict__) and ('_dbus_interface' in cls.__dict__[method_name].__dict__) and (cls.__dict__[method_name]._dbus_interface == dbus_interface):
                parent_method = cls.__dict__[method_name]
                successful = True
                break
    else:
        for cls in self.__class__.__mro__:
            if not candidate_class and method_name in cls.__dict__:
                candidate_class = cls
            if candidate_class and method_name in cls.__dict__ and ('_dbus_is_method' in cls.__dict__[method_name].__dict__):
                parent_method = cls.__dict__[method_name]
                successful = True
                break
    if successful:
        return (candidate_class.__dict__[method_name], parent_method)
    elif dbus_interface:
        raise UnknownMethodException('%s is not a valid method of interface %s' % (method_name, dbus_interface))
    else:
        raise UnknownMethodException('%s is not a valid method' % method_name)