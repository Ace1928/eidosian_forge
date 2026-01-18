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
def test_onlyTopModules(self) -> None:
    """
        Verify that the iterModules API will only return top-level modules and
        packages, not submodules or subpackages.
        """
    for module in modules.iterModules():
        self.assertFalse('.' in module.name, 'no nested modules should be returned from iterModules: %r' % module.filePath)