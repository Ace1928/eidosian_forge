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
def _reflect_on_signal(cls, func):
    args = func._dbus_args
    if func._dbus_signature:
        sig = tuple(Signature(func._dbus_signature))
    else:
        sig = _VariantSignature()
    reflection_data = '    <signal name="%s">\n' % func.__name__
    for pair in zip(sig, args):
        reflection_data = reflection_data + '      <arg type="%s" name="%s" />\n' % pair
    reflection_data = reflection_data + '    </signal>\n'
    return reflection_data