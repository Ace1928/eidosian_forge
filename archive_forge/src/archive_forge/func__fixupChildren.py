import sys, os, time, io, re, traceback, warnings, weakref, collections.abc
from types import GenericAlias
from string import Template
from string import Formatter as StrFormatter
import threading
import atexit
def _fixupChildren(self, ph, alogger):
    """
        Ensure that children of the placeholder ph are connected to the
        specified logger.
        """
    name = alogger.name
    namelen = len(name)
    for c in ph.loggerMap.keys():
        if c.parent.name[:namelen] != name:
            alogger.parent = c.parent
            c.parent = alogger