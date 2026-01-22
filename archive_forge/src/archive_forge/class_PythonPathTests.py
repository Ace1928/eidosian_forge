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
class PythonPathTests(TestCase):
    """
    Tests for the class which provides the implementation for all of the
    public API of L{twisted.python.modules}, L{PythonPath}.
    """

    def test_unhandledImporter(self) -> None:
        """
        Make sure that the behavior when encountering an unknown importer
        type is not catastrophic failure.
        """

        class SecretImporter:
            pass

        def hook(name: object) -> SecretImporter:
            return SecretImporter()
        syspath = ['example/path']
        sysmodules: dict[str, ModuleType] = {}
        syshooks = [hook]
        syscache: dict[str, PathEntryFinder | None] = {}

        def sysloader(name: object) -> None:
            return None
        space = modules.PythonPath(syspath, sysmodules, syshooks, syscache, sysloader)
        entries = list(space.iterEntries())
        self.assertEqual(len(entries), 1)
        self.assertRaises(KeyError, lambda: entries[0]['module'])

    def test_inconsistentImporterCache(self) -> None:
        """
        If the path a module loaded with L{PythonPath.__getitem__} is not
        present in the path importer cache, a warning is emitted, but the
        L{PythonModule} is returned as usual.
        """
        space = modules.PythonPath([], sys.modules, [], {})
        thisModule = space[__name__]
        warnings = self.flushWarnings([self.test_inconsistentImporterCache])
        self.assertEqual(warnings[0]['category'], UserWarning)
        self.assertEqual(warnings[0]['message'], FilePath(twisted.__file__).parent().dirname() + ' (for module ' + __name__ + ') not in path importer cache (PEP 302 violation - check your local configuration).')
        self.assertEqual(len(warnings), 1)
        self.assertEqual(thisModule.name, __name__)

    def test_containsModule(self) -> None:
        """
        L{PythonPath} implements the C{in} operator so that when it is the
        right-hand argument and the name of a module which exists on that
        L{PythonPath} is the left-hand argument, the result is C{True}.
        """
        thePath = modules.PythonPath()
        self.assertIn('os', thePath)

    def test_doesntContainModule(self) -> None:
        """
        L{PythonPath} implements the C{in} operator so that when it is the
        right-hand argument and the name of a module which does not exist on
        that L{PythonPath} is the left-hand argument, the result is C{False}.
        """
        thePath = modules.PythonPath()
        self.assertNotIn('bogusModule', thePath)