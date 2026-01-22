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
class WithoutModuleTests(unittest.SynchronousTestCase):
    """
    Test the C{without-module} flag.
    """

    def setUp(self) -> None:
        """
        Create a L{trial.Options} object to be used in the tests, and save
        C{sys.modules}.
        """
        self.config = trial.Options()
        self.savedModules = dict(sys.modules)

    def tearDown(self) -> None:
        """
        Restore C{sys.modules}.
        """
        for module in ('imaplib', 'smtplib'):
            if module in self.savedModules:
                sys.modules[module] = self.savedModules[module]
            else:
                sys.modules.pop(module, None)

    def _checkSMTP(self) -> object:
        """
        Try to import the C{smtplib} module, and return it.
        """
        import smtplib
        return smtplib

    def _checkIMAP(self) -> object:
        """
        Try to import the C{imaplib} module, and return it.
        """
        import imaplib
        return imaplib

    def test_disableOneModule(self) -> None:
        """
        Check that after disabling a module, it can't be imported anymore.
        """
        self.config.parseOptions(['--without-module', 'smtplib'])
        self.assertRaises(ImportError, self._checkSMTP)
        del sys.modules['smtplib']
        self.assertIsInstance(self._checkSMTP(), types.ModuleType)

    def test_disableMultipleModules(self) -> None:
        """
        Check that several modules can be disabled at once.
        """
        self.config.parseOptions(['--without-module', 'smtplib,imaplib'])
        self.assertRaises(ImportError, self._checkSMTP)
        self.assertRaises(ImportError, self._checkIMAP)
        del sys.modules['smtplib']
        del sys.modules['imaplib']
        self.assertIsInstance(self._checkSMTP(), types.ModuleType)
        self.assertIsInstance(self._checkIMAP(), types.ModuleType)

    def test_disableAlreadyImportedModule(self) -> None:
        """
        Disabling an already imported module should produce a warning.
        """
        self.assertIsInstance(self._checkSMTP(), types.ModuleType)
        self.assertWarns(RuntimeWarning, "Module 'smtplib' already imported, disabling anyway.", trial.__file__, self.config.parseOptions, ['--without-module', 'smtplib'])
        self.assertRaises(ImportError, self._checkSMTP)