from __future__ import annotations
import locale
import os
import sys
from io import StringIO
from typing import Generator
from zope.interface import implementer
from hamcrest import assert_that, equal_to
from twisted.internet.base import DelayedCall
from twisted.internet.interfaces import IProcessTransport
from twisted.python import filepath
from twisted.python.failure import Failure
from twisted.trial import util
from twisted.trial.unittest import SynchronousTestCase
from twisted.trial.util import (
class MktempTests(SynchronousTestCase):
    """
    Tests for L{TestCase.mktemp}, a helper function for creating temporary file
    or directory names.
    """

    def test_name(self) -> None:
        """
        The path name returned by C{mktemp} is directly beneath a directory
        which identifies the test method which created the name.
        """
        name = self.mktemp()
        dirs = os.path.dirname(name).split(os.sep)[:-1]
        self.assertEqual(dirs, ['twisted.trial.test.test_util', 'MktempTests', 'test_name'])

    def test_unique(self) -> None:
        """
        Repeated calls to C{mktemp} return different values.
        """
        name = self.mktemp()
        self.assertNotEqual(name, self.mktemp())

    def test_created(self) -> None:
        """
        The directory part of the path name returned by C{mktemp} exists.
        """
        name = self.mktemp()
        dirname = os.path.dirname(name)
        self.assertTrue(os.path.exists(dirname))
        self.assertFalse(os.path.exists(name))

    def test_location(self) -> None:
        """
        The path returned by C{mktemp} is beneath the current working directory.
        """
        path = os.path.abspath(self.mktemp())
        self.assertTrue(path.startswith(os.getcwd()))