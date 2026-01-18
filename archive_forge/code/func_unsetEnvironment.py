from __future__ import annotations
import compileall
import errno
import functools
import os
import sys
import time
from importlib import invalidate_caches as invalidateImportCaches
from types import ModuleType
from typing import Callable, TypedDict, TypeVar
from zope.interface import Interface
from twisted import plugin
from twisted.python.filepath import FilePath
from twisted.python.log import EventDict, addObserver, removeObserver, textFromEventDict
from twisted.trial import unittest
from twisted.plugin import pluginPackagePaths
def unsetEnvironment(self) -> None:
    """
        Change the Python environment back to what it was before the test was
        started.
        """
    invalidateImportCaches()
    sys.modules.clear()
    sys.modules.update(self.savedModules)
    sys.path[:] = self.savedPath