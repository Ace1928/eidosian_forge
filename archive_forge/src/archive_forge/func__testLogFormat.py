from sys import stderr, stdout
from typing import Callable, Dict, List, Optional, TextIO, Tuple
import twisted.trial.unittest
from twisted.copyright import version
from twisted.internet import reactor
from twisted.internet.interfaces import IReactorCore
from twisted.internet.testing import MemoryReactor
from twisted.logger import (
from twisted.python.usage import UsageError
from ...reactors import NoSuchReactor
from ...runner._exit import ExitStatus
from ...runner.test.test_runner import DummyExit
from ...service import ServiceMaker
from ...twist import _options
from .._options import TwistOptions
def _testLogFormat(self, format: str, expectedObserverFactory: Callable[[TextIO], FileLogObserver]) -> None:
    """
        Set log file format and check the selected observer factory.

        @param format: The format of the file.
        @param expectedObserverFactory: The expected observer factory.
        """
    options = TwistOptions()
    options.opt_log_format(format)
    self.assertIdentical(options['fileLogObserverFactory'], expectedObserverFactory)
    self.assertEqual(options['logFormat'], format)