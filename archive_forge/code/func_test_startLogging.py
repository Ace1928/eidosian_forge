import errno
from io import StringIO
from signal import SIGTERM
from types import TracebackType
from typing import Any, Iterable, List, Optional, TextIO, Tuple, Type, Union, cast
from attr import Factory, attrib, attrs
import twisted.trial.unittest
from twisted.internet.testing import MemoryReactor
from twisted.logger import (
from twisted.python.filepath import FilePath
from ...runner import _runner
from .._exit import ExitStatus
from .._pidfile import NonePIDFile, PIDFile
from .._runner import Runner
def test_startLogging(self) -> None:
    """
        L{Runner.startLogging} sets up a filtering observer with a log level
        predicate set to the given log level that contains a file observer of
        the given type which writes to the given file.
        """
    logFile = StringIO()

    class LogBeginner:
        observers: List[ILogObserver] = []

        def beginLoggingTo(self, observers: Iterable[ILogObserver]) -> None:
            LogBeginner.observers = list(observers)
    self.patch(_runner, 'globalLogBeginner', LogBeginner())

    class MockFilteringLogObserver(FilteringLogObserver):
        observer: Optional[ILogObserver] = None
        predicates: List[LogLevelFilterPredicate] = []

        def __init__(self, observer: ILogObserver, predicates: Iterable[LogLevelFilterPredicate], negativeObserver: ILogObserver=cast(ILogObserver, lambda event: None)) -> None:
            MockFilteringLogObserver.observer = observer
            MockFilteringLogObserver.predicates = list(predicates)
            FilteringLogObserver.__init__(self, observer, predicates, negativeObserver)
    self.patch(_runner, 'FilteringLogObserver', MockFilteringLogObserver)

    class MockFileLogObserver(FileLogObserver):
        outFile: Optional[TextIO] = None

        def __init__(self, outFile: TextIO) -> None:
            MockFileLogObserver.outFile = outFile
            FileLogObserver.__init__(self, outFile, str)
    runner = Runner(reactor=MemoryReactor(), defaultLogLevel=LogLevel.critical, logFile=logFile, fileLogObserverFactory=MockFileLogObserver)
    runner.startLogging()
    self.assertEqual(len(LogBeginner.observers), 1)
    self.assertIsInstance(LogBeginner.observers[0], FilteringLogObserver)
    self.assertEqual(len(MockFilteringLogObserver.predicates), 1)
    self.assertIsInstance(MockFilteringLogObserver.predicates[0], LogLevelFilterPredicate)
    self.assertIdentical(MockFilteringLogObserver.predicates[0].defaultLogLevel, LogLevel.critical)
    observer = cast(MockFileLogObserver, MockFilteringLogObserver.observer)
    self.assertIsInstance(observer, MockFileLogObserver)
    self.assertIdentical(observer.outFile, logFile)