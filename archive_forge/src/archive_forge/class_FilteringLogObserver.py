from functools import partial
from typing import Dict, Iterable
from zope.interface import Interface, implementer
from constantly import NamedConstant, Names
from ._interfaces import ILogObserver, LogEvent
from ._levels import InvalidLogLevelError, LogLevel
from ._observer import bitbucketLogObserver
@implementer(ILogObserver)
class FilteringLogObserver:
    """
    L{ILogObserver} that wraps another L{ILogObserver}, but filters out events
    based on applying a series of L{ILogFilterPredicate}s.
    """

    def __init__(self, observer: ILogObserver, predicates: Iterable[ILogFilterPredicate], negativeObserver: ILogObserver=bitbucketLogObserver) -> None:
        """
        @param observer: An observer to which this observer will forward
            events when C{predictates} yield a positive result.
        @param predicates: Predicates to apply to events before forwarding to
            the wrapped observer.
        @param negativeObserver: An observer to which this observer will
            forward events when C{predictates} yield a negative result.
        """
        self._observer = observer
        self._shouldLogEvent = partial(shouldLogEvent, list(predicates))
        self._negativeObserver = negativeObserver

    def __call__(self, event: LogEvent) -> None:
        """
        Forward to next observer if predicate allows it.
        """
        if self._shouldLogEvent(event):
            if 'log_trace' in event:
                event['log_trace'].append((self, self._observer))
            self._observer(event)
        else:
            self._negativeObserver(event)