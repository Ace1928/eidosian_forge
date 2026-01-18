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
def test_plugins(self) -> None:
    """
        L{plugin.getPlugins} should return the list of plugins matching the
        specified interface (here, L{ITestPlugin2}), and these plugins
        should be instances of classes with a C{test} method, to be sure
        L{plugin.getPlugins} load classes correctly.
        """
    plugins = list(plugin.getPlugins(ITestPlugin2, self.module))
    self.assertEqual(len(plugins), 2)
    names = ['AnotherTestPlugin', 'ThirdTestPlugin']
    for p in plugins:
        names.remove(p.__name__)
        p.test()