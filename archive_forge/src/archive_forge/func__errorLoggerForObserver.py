from typing import Callable, Optional
from zope.interface import implementer
from twisted.python.failure import Failure
from ._interfaces import ILogObserver, LogEvent
from ._logger import Logger
def _errorLoggerForObserver(self, observer: ILogObserver) -> Logger:
    """
        Create an error-logger based on this logger, which does not contain the
        given bad observer.

        @param observer: The observer which previously had an error.

        @return: A L{Logger} without the given observer.
        """
    errorPublisher = LogPublisher(*(obs for obs in self._observers if obs is not observer))
    return Logger(observer=errorPublisher)