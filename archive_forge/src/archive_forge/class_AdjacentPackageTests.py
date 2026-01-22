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
class AdjacentPackageTests(unittest.TestCase):
    """
    Tests for the behavior of the plugin system when there are multiple
    installed copies of the package containing the plugins being loaded.
    """

    def setUp(self) -> None:
        """
        Save the elements of C{sys.path} and the items of C{sys.modules}.
        """
        self.originalPath = sys.path[:]
        self.savedModules = sys.modules.copy()

    def tearDown(self) -> None:
        """
        Restore C{sys.path} and C{sys.modules} to their original values.
        """
        sys.path[:] = self.originalPath
        sys.modules.clear()
        sys.modules.update(self.savedModules)

    def createDummyPackage(self, root: FilePath[str], name: str, pluginName: str) -> FilePath[str]:
        """
        Create a directory containing a Python package named I{dummy} with a
        I{plugins} subpackage.

        @type root: L{FilePath}
        @param root: The directory in which to create the hierarchy.

        @type name: C{str}
        @param name: The name of the directory to create which will contain
            the package.

        @type pluginName: C{str}
        @param pluginName: The name of a module to create in the
            I{dummy.plugins} package.

        @rtype: L{FilePath}
        @return: The directory which was created to contain the I{dummy}
            package.
        """
        directory = root.child(name)
        package = directory.child('dummy')
        package.makedirs()
        package.child('__init__.py').setContent(b'')
        plugins = package.child('plugins')
        plugins.makedirs()
        plugins.child('__init__.py').setContent(pluginInitFile)
        pluginModule = plugins.child(pluginName + '.py')
        pluginModule.setContent(pluginFileContents(name))
        return directory

    def test_hiddenPackageSamePluginModuleNameObscured(self) -> None:
        """
        Only plugins from the first package in sys.path should be returned by
        getPlugins in the case where there are two Python packages by the same
        name installed, each with a plugin module by a single name.
        """
        root = FilePath(self.mktemp())
        root.makedirs()
        firstDirectory = self.createDummyPackage(root, 'first', 'someplugin')
        secondDirectory = self.createDummyPackage(root, 'second', 'someplugin')
        sys.path.append(firstDirectory.path)
        sys.path.append(secondDirectory.path)
        import dummy.plugins
        plugins = list(plugin.getPlugins(ITestPlugin, dummy.plugins))
        self.assertEqual(['first'], [p.__name__ for p in plugins])

    def test_hiddenPackageDifferentPluginModuleNameObscured(self) -> None:
        """
        Plugins from the first package in sys.path should be returned by
        getPlugins in the case where there are two Python packages by the same
        name installed, each with a plugin module by a different name.
        """
        root = FilePath(self.mktemp())
        root.makedirs()
        firstDirectory = self.createDummyPackage(root, 'first', 'thisplugin')
        secondDirectory = self.createDummyPackage(root, 'second', 'thatplugin')
        sys.path.append(firstDirectory.path)
        sys.path.append(secondDirectory.path)
        import dummy.plugins
        plugins = list(plugin.getPlugins(ITestPlugin, dummy.plugins))
        self.assertEqual(['first'], [p.__name__ for p in plugins])