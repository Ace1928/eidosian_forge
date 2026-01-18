from sys import stdout
from typing import Any, Dict, List
import twisted.trial.unittest
from twisted.internet.interfaces import IReactorCore
from twisted.internet.testing import MemoryReactor
from twisted.logger import LogLevel, jsonFileLogObserver
from twisted.test.test_twistd import SignalCapturingMemoryReactor
from ...runner._exit import ExitStatus
from ...runner._runner import Runner
from ...runner.test.test_runner import DummyExit
from ...service import IService, MultiService
from ...twist import _twist
from .._options import TwistOptions
from .._twist import Twist
def test_optionsInvalidArguments(self) -> None:
    """
        L{Twist.options} given invalid arguments exits with
        L{ExitStatus.EX_USAGE} and an error/usage message.
        """
    self.patchExit()
    Twist.options(['twist', '--bogus-bagels'])
    self.assertIdentical(self.exit.status, ExitStatus.EX_USAGE)
    self.assertIsNotNone(self.exit.message)
    self.assertTrue(self.exit.message.startswith('Error: '))
    self.assertTrue(self.exit.message.endswith(f'\n\n{TwistOptions()}'))