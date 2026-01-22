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
class InstallationTests(SynchronousTestCase):
    """
    Tests for actual installation of the reactor.
    """

    def test_install(self) -> None:
        """
        L{install} installs a reactor.
        """
        with NoReactor():
            install()
            self.assertIn('twisted.internet.reactor', sys.modules)

    def test_reactor(self) -> None:
        """
        Importing L{twisted.internet.reactor} installs the default reactor if
        none is installed.
        """
        installed: list[bool] = []

        def installer() -> object:
            installed.append(True)
            return install()
        self.patch(default, 'install', installer)
        with NoReactor():
            from twisted.internet import reactor
            self.assertTrue(IReactorCore.providedBy(reactor))
            self.assertEqual(installed, [True])