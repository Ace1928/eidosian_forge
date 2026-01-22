import sys, os, time, io, re, traceback, warnings, weakref, collections.abc
from types import GenericAlias
from string import Template
from string import Formatter as StrFormatter
import threading
import atexit
def _removeHandlerRef(wr):
    """
    Remove a handler reference from the internal cleanup list.
    """
    acquire, release, handlers = (_acquireLock, _releaseLock, _handlerList)
    if acquire and release and handlers:
        acquire()
        try:
            handlers.remove(wr)
        except ValueError:
            pass
        finally:
            release()