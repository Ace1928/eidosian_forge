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
class MockFilteringLogObserver(FilteringLogObserver):
    observer: Optional[ILogObserver] = None
    predicates: List[LogLevelFilterPredicate] = []

    def __init__(self, observer: ILogObserver, predicates: Iterable[LogLevelFilterPredicate], negativeObserver: ILogObserver=cast(ILogObserver, lambda event: None)) -> None:
        MockFilteringLogObserver.observer = observer
        MockFilteringLogObserver.predicates = list(predicates)
        FilteringLogObserver.__init__(self, observer, predicates, negativeObserver)