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