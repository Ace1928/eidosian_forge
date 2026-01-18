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
def test_pluginPackagesExcluded(self) -> None:
    """
        L{plugin.pluginPackagePaths} should exclude directories which are
        Python packages.  The only allowed plugin package (the only one
        associated with a I{dummy} package which Python will allow to be
        imported) will already be known to the caller of
        L{plugin.pluginPackagePaths} and will most commonly already be in
        the C{__path__} they are about to mutate.
        """
    root = FilePath(self.mktemp())
    foo = root.child('foo').child('dummy').child('plugins')
    foo.makedirs()
    foo.child('__init__.py').setContent(b'')
    sys.path = [root.child('foo').path, root.child('bar').path]
    self.assertEqual(plugin.pluginPackagePaths('dummy.plugins'), [root.child('bar').child('dummy').child('plugins').path])