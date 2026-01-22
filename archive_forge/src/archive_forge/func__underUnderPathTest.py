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
def _underUnderPathTest(self, doImport: bool=True) -> None:
    moddir2 = self.mktemp()
    fpmd = FilePath(moddir2)
    fpmd.createDirectory()
    fpmd.child('foozle.py').setContent(b'x = 123\n')
    self.packagePath.child('__init__.py').setContent(networkString(f'__path__.append({repr(moddir2)})\n'))
    self._setupSysPath()
    modinfo = modules.getModule(self.packageName)
    self.assertEqual(self.findByIteration(self.packageName + '.foozle', modinfo, importPackages=doImport), modinfo['foozle'])
    self.assertEqual(modinfo['foozle'].load().x, 123)