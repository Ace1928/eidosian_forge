import glob
import inspect
import logging
import os
import platform
import importlib.util
import sys
from . import envvar
from .dependencies import ctypes
from .deprecation import deprecated, relocated_module_attribute
class PathData(object):
    """An object for storing and managing a :py:class:`PathManager` path"""

    def __init__(self, manager, name):
        self._mngr = manager
        self._registered_name = name
        self._path = None
        self._path_override = None
        self._status = None

    def path(self):
        """Return the full, normalized path to the registered path entry.

        If the object is not found (or was marked "disabled"),
        ``path()`` returns None.

        """
        if self._status is None:
            if self._path_override:
                target = self._path_override
            else:
                target = self._registered_name
            tmp = self._mngr._find(target, pathlist=self._mngr.pathlist)
            self._path = tmp if tmp else self._path_override
            self._status = bool(tmp)
        return self._path

    def set_path(self, value):
        self._path_override = value
        self.rehash()
        if not self._status:
            logging.getLogger('pyomo.common').warning("explicitly setting the path for '%s' to an invalid object or nonexistent location ('%s')" % (self._registered_name, value))

    @deprecated('get_path() is deprecated; use pyomo.common.Executable(name).path()', version='5.6.2')
    def get_path(self):
        return self.path()

    def disable(self):
        """Disable this path entry

        This method "disables" this path entry by marking it as "not
        found".  Disabled entries return False for `available()` and
        None for `path()`.  The disabled status will persist until the
        next call to `rehash()`.

        """
        self._status = False
        self._path = None

    def available(self):
        """Returns True if the registered path is available.

        Entries are available if the object was found found in the
        search locations and has not been explicitly disabled.

        """
        if self._status is None:
            self.path()
        return self._status

    def rehash(self):
        """Requery the location of this path entry

        This method derives its name from the csh command of the same
        name, which rebuilds the hash table of executables reachable
        through the PATH.

        """
        self._status = None
        self.path()

    def __nonzero__(self):
        """Alias for ``available()``."""
        return self.available()
    __bool__ = __nonzero__

    def __str__(self):
        ans = self.path()
        if not ans:
            return ''
        return ans