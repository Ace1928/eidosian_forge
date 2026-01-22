import warnings
import sys
import socket
from .._ossighelper import wakeup_on_signal, register_sigint_fallback
from ..module import get_introspection_module
from .._gi import (variant_type_from_string, source_new,
from ..overrides import override, deprecated, deprecated_attr
from gi import PyGIDeprecationWarning, version_info
from gi import _option as option
from gi import _gi
from gi._error import GError
def _child_watch_add_get_args(priority_or_pid, pid_or_callback, *args, **kwargs):
    user_data = []
    if callable(pid_or_callback):
        warnings.warn('Calling child_watch_add without priority as first argument is deprecated', PyGIDeprecationWarning)
        pid = priority_or_pid
        callback = pid_or_callback
        if len(args) == 0:
            priority = kwargs.get('priority', GLib.PRIORITY_DEFAULT)
        elif len(args) == 1:
            user_data = args
            priority = kwargs.get('priority', GLib.PRIORITY_DEFAULT)
        elif len(args) == 2:
            user_data = [args[0]]
            priority = args[1]
        else:
            raise TypeError('expected at most 4 positional arguments')
    else:
        priority = priority_or_pid
        pid = pid_or_callback
        if 'function' in kwargs:
            callback = kwargs['function']
            user_data = args
        elif len(args) > 0 and callable(args[0]):
            callback = args[0]
            user_data = args[1:]
        else:
            raise TypeError('expected callback as third argument')
    if 'data' in kwargs:
        if user_data:
            raise TypeError('got multiple values for "data" argument')
        user_data = (kwargs['data'],)
    return (priority, pid, callback, user_data)