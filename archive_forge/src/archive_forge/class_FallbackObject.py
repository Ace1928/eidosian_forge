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
class FallbackObject(Object):
    """An object that implements an entire subtree of the object-path
    tree.

    :Since: 0.82.0
    """
    SUPPORTS_MULTIPLE_OBJECT_PATHS = True

    def __init__(self, conn=None, object_path=None):
        """Constructor.

        Note that the superclass' ``bus_name`` __init__ argument is not
        supported here.

        :Parameters:
            `conn` : dbus.connection.Connection or None
                The connection on which to export this object. If this is not
                None, an `object_path` must also be provided.

                If None, the object is not initially available on any
                Connection.

            `object_path` : str or None
                A D-Bus object path at which to make this Object available
                immediately. If this is not None, a `conn` must also be
                provided.

                This object will implements all object-paths in the subtree
                starting at this object-path, except where a more specific
                object has been added.
        """
        super(FallbackObject, self).__init__()
        self._fallback = True
        if conn is None:
            if object_path is not None:
                raise TypeError('If object_path is given, conn is required')
        elif object_path is None:
            raise TypeError('If conn is given, object_path is required')
        else:
            self.add_to_connection(conn, object_path)