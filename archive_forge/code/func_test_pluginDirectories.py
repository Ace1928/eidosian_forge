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
def test_pluginDirectories(self) -> None:
    """
        L{plugin.pluginPackagePaths} should return a list containing each
        directory in C{sys.path} with a suffix based on the supplied package
        name.
        """
    foo = FilePath('foo')
    bar = FilePath('bar')
    sys.path = [foo.path, bar.path]
    self.assertEqual(plugin.pluginPackagePaths('dummy.plugins'), [foo.child('dummy').child('plugins').path, bar.child('dummy').child('plugins').path])