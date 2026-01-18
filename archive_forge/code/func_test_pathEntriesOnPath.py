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
def test_pathEntriesOnPath(self) -> None:
    """
        Verify that path entries discovered via module loading are, in fact, on
        sys.path somewhere.
        """
    for n in ['os', 'twisted', 'twisted.python', 'twisted.python.reflect']:
        self.failUnlessIn(modules.getModule(n).pathEntry.filePath.path, sys.path)