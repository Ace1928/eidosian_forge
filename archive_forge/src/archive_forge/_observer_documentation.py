from typing import Callable, Optional
from zope.interface import implementer
from twisted.python.failure import Failure
from ._interfaces import ILogObserver, LogEvent
from ._logger import Logger

                Add tracing information for an observer.

                @param observer: an observer being forwarded to
                