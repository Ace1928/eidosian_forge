import gireactor or gtk3reactor for GObject Introspection based applications,
import sys
from typing import Any, Callable, Dict, Set
from zope.interface import implementer
from twisted.internet import posixbase
from twisted.internet.abstract import FileDescriptor
from twisted.internet.interfaces import IReactorFDSet, IReadDescriptor, IWriteDescriptor
from twisted.python import log
from twisted.python.monkey import MonkeyPatcher
from ._signals import _IWaker, _UnixWaker
def _signalGlue():
    """
    Integrate glib's wakeup file descriptor usage and our own.

    Python supports only one wakeup file descriptor at a time and both Twisted
    and glib want to use it.

    This is a context manager that can be wrapped around the whole glib
    reactor main loop which makes our signal handling work with glib's signal
    handling.
    """
    from gi import _ossighelper as signalGlue
    patcher = MonkeyPatcher()
    patcher.addPatch(signalGlue, '_wakeup_fd_is_active', True)
    return patcher