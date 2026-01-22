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
class TwistExitTests(twisted.trial.unittest.TestCase):
    """
    Tests to verify that the Twist script takes the expected actions related
    to signals and the reactor.
    """

    def setUp(self) -> None:
        self.exitWithSignalCalled = False

        def fakeExitWithSignal(sig: int) -> None:
            """
            Fake to capture whether L{twisted.application._exitWithSignal
            was called.

            @param sig: Signal value
            @type sig: C{int}
            """
            self.exitWithSignalCalled = True
        self.patch(_twist, '_exitWithSignal', fakeExitWithSignal)

        def startLogging(_: Runner) -> None:
            """
            Prevent Runner from adding new log observers or other
            tests outside this module will fail.

            @param _: Unused self param
            """
        self.patch(Runner, 'startLogging', startLogging)

    def test_twistReactorDoesntExitWithSignal(self) -> None:
        """
        _exitWithSignal is not called if the reactor's _exitSignal attribute
        is zero.
        """
        reactor = SignalCapturingMemoryReactor()
        reactor._exitSignal = None
        options = TwistOptions()
        options['reactor'] = reactor
        options['fileLogObserverFactory'] = jsonFileLogObserver
        Twist.run(options)
        self.assertFalse(self.exitWithSignalCalled)

    def test_twistReactorHasNoExitSignalAttr(self) -> None:
        """
        _exitWithSignal is not called if the runner's reactor does not
        implement L{twisted.internet.interfaces._ISupportsExitSignalCapturing}
        """
        reactor = MemoryReactor()
        options = TwistOptions()
        options['reactor'] = reactor
        options['fileLogObserverFactory'] = jsonFileLogObserver
        Twist.run(options)
        self.assertFalse(self.exitWithSignalCalled)

    def test_twistReactorExitsWithSignal(self) -> None:
        """
        _exitWithSignal is called if the runner's reactor exits due
        to a signal.
        """
        reactor = SignalCapturingMemoryReactor()
        reactor._exitSignal = 2
        options = TwistOptions()
        options['reactor'] = reactor
        options['fileLogObserverFactory'] = jsonFileLogObserver
        Twist.run(options)
        self.assertTrue(self.exitWithSignalCalled)