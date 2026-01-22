from __future__ import annotations
import compileall
import itertools
import sys
import zipfile
from importlib.abc import PathEntryFinder
from types import ModuleType
from typing import Any, Generator
from typing_extensions import Protocol
import twisted
from twisted.python import modules
from twisted.python.compat import networkString
from twisted.python.filepath import FilePath
from twisted.python.reflect import namedAny
from twisted.python.test.modules_helpers import TwistedModulesMixin
from twisted.python.test.test_zippath import zipit
from twisted.trial.unittest import TestCase
class BasicTests(TwistedModulesTestCase):

    def test_namespacedPackages(self) -> None:
        """
        Duplicate packages are not yielded when iterating over namespace
        packages.
        """
        __import__('pkgutil')
        namespaceBoilerplate = b'import pkgutil; __path__ = pkgutil.extend_path(__path__, __name__)'
        entry = self.pathEntryWithOnePackage()
        testPackagePath = entry.child('test_package')
        testPackagePath.child('__init__.py').setContent(namespaceBoilerplate)
        nestedEntry = testPackagePath.child('nested_package')
        nestedEntry.makedirs()
        nestedEntry.child('__init__.py').setContent(namespaceBoilerplate)
        nestedEntry.child('module.py').setContent(b'')
        anotherEntry = self.pathEntryWithOnePackage()
        anotherPackagePath = anotherEntry.child('test_package')
        anotherPackagePath.child('__init__.py').setContent(namespaceBoilerplate)
        anotherNestedEntry = anotherPackagePath.child('nested_package')
        anotherNestedEntry.makedirs()
        anotherNestedEntry.child('__init__.py').setContent(namespaceBoilerplate)
        anotherNestedEntry.child('module2.py').setContent(b'')
        self.replaceSysPath([entry.path, anotherEntry.path])
        module = modules.getModule('test_package')
        try:
            walkedNames = [mod.name for mod in module.walkModules(importPackages=True)]
        finally:
            for module in list(sys.modules.keys()):
                if module.startswith('test_package'):
                    del sys.modules[module]
        expected = ['test_package', 'test_package.nested_package', 'test_package.nested_package.module', 'test_package.nested_package.module2']
        self.assertEqual(walkedNames, expected)

    def test_unimportablePackageGetItem(self) -> None:
        """
        If a package has been explicitly forbidden from importing by setting a
        L{None} key in sys.modules under its name,
        L{modules.PythonPath.__getitem__} should still be able to retrieve an
        unloaded L{modules.PythonModule} for that package.
        """
        shouldNotLoad: list[str] = []
        path = modules.PythonPath(sysPath=[self.pathEntryWithOnePackage().path], moduleLoader=shouldNotLoad.append, importerCache={}, sysPathHooks={}, moduleDict={'test_package': None})
        self.assertEqual(shouldNotLoad, [])
        self.assertFalse(path['test_package'].isLoaded())

    def test_unimportablePackageWalkModules(self) -> None:
        """
        If a package has been explicitly forbidden from importing by setting a
        L{None} key in sys.modules under its name, L{modules.walkModules} should
        still be able to retrieve an unloaded L{modules.PythonModule} for that
        package.
        """
        existentPath = self.pathEntryWithOnePackage()
        self.replaceSysPath([existentPath.path])
        self.replaceSysModules({'test_package': None})
        walked = list(modules.walkModules())
        self.assertEqual([m.name for m in walked], ['test_package'])
        self.assertFalse(walked[0].isLoaded())

    def test_nonexistentPaths(self) -> None:
        """
        Verify that L{modules.walkModules} ignores entries in sys.path which
        do not exist in the filesystem.
        """
        existentPath = self.pathEntryWithOnePackage()
        nonexistentPath = FilePath(self.mktemp())
        self.assertFalse(nonexistentPath.exists())
        self.replaceSysPath([existentPath.path])
        expected = [modules.getModule('test_package')]
        beforeModules = list(modules.walkModules())
        sys.path.append(nonexistentPath.path)
        afterModules = list(modules.walkModules())
        self.assertEqual(beforeModules, expected)
        self.assertEqual(afterModules, expected)

    def test_nonDirectoryPaths(self) -> None:
        """
        Verify that L{modules.walkModules} ignores entries in sys.path which
        refer to regular files in the filesystem.
        """
        existentPath = self.pathEntryWithOnePackage()
        nonDirectoryPath = FilePath(self.mktemp())
        self.assertFalse(nonDirectoryPath.exists())
        nonDirectoryPath.setContent(b'zip file or whatever\n')
        self.replaceSysPath([existentPath.path])
        beforeModules = list(modules.walkModules())
        sys.path.append(nonDirectoryPath.path)
        afterModules = list(modules.walkModules())
        self.assertEqual(beforeModules, afterModules)

    def test_twistedShowsUp(self) -> None:
        """
        Scrounge around in the top-level module namespace and make sure that
        Twisted shows up, and that the module thusly obtained is the same as
        the module that we find when we look for it explicitly by name.
        """
        self.assertEqual(modules.getModule('twisted'), self.findByIteration('twisted'))

    def test_dottedNames(self) -> None:
        """
        Verify that the walkModules APIs will give us back subpackages, not just
        subpackages.
        """
        self.assertEqual(modules.getModule('twisted.python'), self.findByIteration('twisted.python', where=modules.getModule('twisted')))

    def test_onlyTopModules(self) -> None:
        """
        Verify that the iterModules API will only return top-level modules and
        packages, not submodules or subpackages.
        """
        for module in modules.iterModules():
            self.assertFalse('.' in module.name, 'no nested modules should be returned from iterModules: %r' % module.filePath)

    def test_loadPackagesAndModules(self) -> None:
        """
        Verify that we can locate and load packages, modules, submodules, and
        subpackages.
        """
        for n in ['os', 'twisted', 'twisted.python', 'twisted.python.reflect']:
            m = namedAny(n)
            self.failUnlessIdentical(modules.getModule(n).load(), m)
            self.failUnlessIdentical(self.findByIteration(n).load(), m)

    def test_pathEntriesOnPath(self) -> None:
        """
        Verify that path entries discovered via module loading are, in fact, on
        sys.path somewhere.
        """
        for n in ['os', 'twisted', 'twisted.python', 'twisted.python.reflect']:
            self.failUnlessIn(modules.getModule(n).pathEntry.filePath.path, sys.path)

    def test_alwaysPreferPy(self) -> None:
        """
        Verify that .py files will always be preferred to .pyc files, regardless of
        directory listing order.
        """
        mypath = FilePath(self.mktemp())
        mypath.createDirectory()
        pp = modules.PythonPath(sysPath=[mypath.path])
        originalSmartPath = pp._smartPath

        def _evilSmartPath(pathName: str) -> Any:
            o = originalSmartPath(pathName)
            originalChildren = o.children

            def evilChildren() -> Any:
                x = list(originalChildren())
                x.sort()
                x.reverse()
                return x
            o.children = evilChildren
            return o
        mypath.child('abcd.py').setContent(b'\n')
        compileall.compile_dir(mypath.path, quiet=True)
        self.assertEqual(len(list(mypath.children())), 2)
        pp._smartPath = _evilSmartPath
        self.assertEqual(pp['abcd'].filePath, mypath.child('abcd.py'))

    def test_packageMissingPath(self) -> None:
        """
        A package can delete its __path__ for some reasons,
        C{modules.PythonPath} should be able to deal with it.
        """
        mypath = FilePath(self.mktemp())
        mypath.createDirectory()
        pp = modules.PythonPath(sysPath=[mypath.path])
        subpath = mypath.child('abcd')
        subpath.createDirectory()
        subpath.child('__init__.py').setContent(b'del __path__\n')
        sys.path.append(mypath.path)
        __import__('abcd')
        try:
            l = list(pp.walkModules())
            self.assertEqual(len(l), 1)
            self.assertEqual(l[0].name, 'abcd')
        finally:
            del sys.modules['abcd']
            sys.path.remove(mypath.path)