from __future__ import annotations
import gc
import re
import sys
import textwrap
import types
from io import StringIO
from typing import List
from hamcrest import assert_that, contains_string
from hypothesis import given
from hypothesis.strategies import sampled_from
from twisted.logger import Logger
from twisted.python import util
from twisted.python.filepath import FilePath, IFilePath
from twisted.python.usage import UsageError
from twisted.scripts import trial
from twisted.trial import unittest
from twisted.trial._dist.disttrial import DistTrialRunner
from twisted.trial._dist.functional import compose
from twisted.trial.runner import (
from twisted.trial.test.test_loader import testNames
from .matchers import fileContents
class SuiteUsedTests(unittest.SynchronousTestCase):
    """
    Check the category of tests suite used by the loader.
    """

    def setUp(self) -> None:
        """
        Create a trial configuration object.
        """
        self.config = trial.Options()

    def test_defaultSuite(self) -> None:
        """
        By default, the loader should use L{DestructiveTestSuite}
        """
        loader = trial._getLoader(self.config)
        self.assertEqual(loader.suiteFactory, DestructiveTestSuite)

    def test_untilFailureSuite(self) -> None:
        """
        The C{until-failure} configuration uses the L{TestSuite} to keep
        instances alive across runs.
        """
        self.config['until-failure'] = True
        loader = trial._getLoader(self.config)
        self.assertEqual(loader.suiteFactory, TestSuite)