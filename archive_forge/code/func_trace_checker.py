import collections
import datetime
import functools
import inspect
import socket
import threading
from oslo_utils import reflection
from oslo_utils import uuidutils
from osprofiler import _utils as utils
from osprofiler import notifier
def trace_checker(attr_name, to_be_wrapped):
    if attr_name.startswith('__'):
        return (False, None)
    if not trace_private and attr_name.startswith('_'):
        return (False, None)
    if isinstance(to_be_wrapped, staticmethod):
        if not trace_static_methods:
            return (False, None)
        return (True, staticmethod)
    if isinstance(to_be_wrapped, classmethod):
        if not trace_class_methods:
            return (False, None)
        return (True, classmethod)
    return (True, None)