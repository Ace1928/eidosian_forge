import sys, os, time, io, re, traceback, warnings, weakref, collections.abc
from types import GenericAlias
from string import Template
from string import Formatter as StrFormatter
import threading
import atexit
def _register_at_fork_reinit_lock(instance):
    _acquireLock()
    try:
        _at_fork_reinit_lock_weakset.add(instance)
    finally:
        _releaseLock()