from __future__ import print_function
import sys
import textwrap
import functools
import itertools
import collections
import gdb
from Cython.Debugger import libpython
class CythonCommand(gdb.Command, CythonBase):
    """
    Base class for Cython commands
    """
    command_class = gdb.COMMAND_NONE

    @classmethod
    def _register(cls, clsname, args, kwargs):
        if not hasattr(cls, 'completer_class'):
            return cls(clsname, cls.command_class, *args, **kwargs)
        else:
            return cls(clsname, cls.command_class, cls.completer_class, *args, **kwargs)

    @classmethod
    def register(cls, *args, **kwargs):
        alias = getattr(cls, 'alias', None)
        if alias:
            cls._register(cls.alias, args, kwargs)
        return cls._register(cls.name, args, kwargs)