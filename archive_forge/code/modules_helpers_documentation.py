import sys
from types import ModuleType
from typing import Iterable, List, Tuple
from twisted.python.filepath import FilePath

        Generate a L{FilePath} with one package, named C{pkgname}, on it, and
        return the L{FilePath} of the path entry.
        