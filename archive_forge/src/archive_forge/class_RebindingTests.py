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
class RebindingTests(PathModificationTests):
    """
    These tests verify that the default path interrogation API works properly
    even when sys.path has been rebound to a different object.
    """

    def _setupSysPath(self) -> None:
        assert not self.pathSetUp
        self.pathSetUp = True
        self.savedSysPath = sys.path
        sys.path = sys.path[:]
        sys.path.append(self.pathExtensionName)

    def tearDown(self) -> None:
        """
        Clean up sys.path by re-binding our original object.
        """
        if self.pathSetUp:
            sys.path = self.savedSysPath