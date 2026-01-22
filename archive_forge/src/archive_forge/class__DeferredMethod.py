import logging
import _dbus_bindings
from dbus._expat_introspect_parser import process_introspection_data
from dbus.exceptions import (
from _dbus_bindings import (
from dbus._compat import is_py2
class _DeferredMethod:
    """A proxy method which will only get called once we have its
    introspection reply.
    """

    def __init__(self, proxy_method, append, block):
        self._proxy_method = proxy_method
        self._method_name = proxy_method._method_name
        self._append = append
        self._block = block

    def __call__(self, *args, **keywords):
        if 'reply_handler' in keywords or keywords.get('ignore_reply', False):
            self._append(self._proxy_method, args, keywords)
            return None
        else:
            self._block()
            return self._proxy_method(*args, **keywords)

    def call_async(self, *args, **keywords):
        self._append(self._proxy_method, args, keywords)