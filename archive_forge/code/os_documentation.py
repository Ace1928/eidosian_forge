import abc
import sys
import stat as st
from _collections_abc import _check_methods
from os.path import (curdir, pardir, sep, pathsep, defpath, extsep, altsep,
from _collections_abc import MutableMapping, Mapping
Add a path to the DLL search path.

        This search path is used when resolving dependencies for imported
        extension modules (the module itself is resolved through sys.path),
        and also by ctypes.

        Remove the directory by calling close() on the returned object or
        using it in a with statement.
        