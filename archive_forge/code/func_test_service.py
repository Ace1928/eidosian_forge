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
def test_service(self) -> None:
    """
        L{Twist.service} returns an L{IService}.
        """
    options = Twist.options(['twist', 'web'])
    service = Twist.service(options.plugins['web'], options.subOptions)
    self.assertTrue(IService.providedBy(service))