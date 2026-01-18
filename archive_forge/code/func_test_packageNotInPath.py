from __future__ import annotations
import os
import sys
import unittest as pyunit
from hashlib import md5
from operator import attrgetter
from types import ModuleType
from typing import TYPE_CHECKING, Callable, Generator
from hamcrest import assert_that, equal_to, has_properties
from hamcrest.core.matcher import Matcher
from twisted.python import filepath, util
from twisted.python.modules import PythonAttribute, PythonModule, getModule
from twisted.python.reflect import ModuleNotFound
from twisted.trial import reporter, runner, unittest
from twisted.trial._asyncrunner import _iterateTests
from twisted.trial.itrial import ITestCase
from twisted.trial.test import packages
from .matchers import after
def test_packageNotInPath(self) -> None:
    """
        If passed the path to a directory which represents a package which
        is not on the import path, L{runner.filenameToModule} returns a
        module object loosely resembling the package defined by that
        directory anyway.
        """
    self.mangleSysPath(self.oldPath)
    package1 = runner.filenameToModule(os.path.join(self.parent, 'goodpackage'))
    self.assertEqual(package1.__name__, 'goodpackage')
    self.cleanUpModules()
    self.mangleSysPath(self.newPath)
    import goodpackage
    self.assertIsNot(package1, goodpackage)
    assert_that(package1, looselyResembles(goodpackage))