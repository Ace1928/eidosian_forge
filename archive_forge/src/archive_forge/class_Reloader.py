from __future__ import print_function, absolute_import
import sys
import struct
import os
from shibokensupport.signature import typing
from shibokensupport.signature.typing import TypeVar, Generic
from shibokensupport.signature.lib.tool import with_metaclass
class Reloader(object):
    """
    Reloder class

    This is a singleton class which provides the update function for the
    shiboken and PySide classes.
    """

    def __init__(self):
        self.sys_module_count = 0

    @staticmethod
    def module_valid(mod):
        if getattr(mod, '__file__', None) and (not os.path.isdir(mod.__file__)):
            ending = os.path.splitext(mod.__file__)[-1]
            return ending not in ('.py', '.pyc', '.pyo', '.pyi')
        return False

    def update(self):
        """
        'update' imports all binary modules which are already in sys.modules.
        The reason is to follow all user imports without introducing new ones.
        This function is called by pyside_type_init to adapt imports
        when the number of imported modules has changed.
        """
        if self.sys_module_count == len(sys.modules):
            return
        self.sys_module_count = len(sys.modules)
        g = globals()
        candidates = list((mod_name for mod_name in sys.modules.copy() if self.module_valid(sys.modules[mod_name])))
        for mod_name in candidates:
            top = __import__(mod_name)
            g[top.__name__] = top
            proc_name = 'init_' + mod_name.replace('.', '_')
            if proc_name in g:
                g.update(g.pop(proc_name)())