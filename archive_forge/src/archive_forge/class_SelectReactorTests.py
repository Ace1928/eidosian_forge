from __future__ import annotations
import select
import sys
from typing import Callable
from twisted.internet import default
from twisted.internet.default import _getInstallFunction, install
from twisted.internet.interfaces import IReactorCore
from twisted.internet.test.test_main import NoReactor
from twisted.python.reflect import requireModule
from twisted.python.runtime import Platform
from twisted.trial.unittest import SynchronousTestCase
class SelectReactorTests(SynchronousTestCase):
    """
    Tests for the cases of L{twisted.internet.default._getInstallFunction}
    in which it picks the select(2)-based reactor.
    """

    def test_osx(self) -> None:
        """
        L{_getInstallFunction} chooses the select reactor on macOS.
        """
        install = _getInstallFunction(osx)
        self.assertEqual(install.__module__, 'twisted.internet.selectreactor')

    def test_windows(self) -> None:
        """
        L{_getInstallFunction} chooses the select reactor on Windows.
        """
        install = _getInstallFunction(windows)
        self.assertEqual(install.__module__, 'twisted.internet.selectreactor')