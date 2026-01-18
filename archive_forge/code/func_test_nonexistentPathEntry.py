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
@_withCacheness
def test_nonexistentPathEntry(self) -> None:
    """
        Test that getCache skips over any entries in a plugin package's
        C{__path__} which do not exist.
        """
    path = self.mktemp()
    self.assertFalse(os.path.exists(path))
    self.module.__path__.append(path)
    try:
        plgs = list(plugin.getPlugins(ITestPlugin, self.module))
        self.assertEqual(len(plgs), 1)
    finally:
        self.module.__path__.remove(path)