import builtins
import functools
import sys
from mako import compat
from mako import exceptions
from mako import util
class _NSAttr:

    def __init__(self, parent):
        self.__parent = parent

    def __getattr__(self, key):
        ns = self.__parent
        while ns:
            if hasattr(ns.module, key):
                return getattr(ns.module, key)
            else:
                ns = ns.inherits
        raise AttributeError(key)