from functools import partial
from typing import Dict, Iterable
from zope.interface import Interface, implementer
from constantly import NamedConstant, Names
from ._interfaces import ILogObserver, LogEvent
from ._levels import InvalidLogLevelError, LogLevel
from ._observer import bitbucketLogObserver
class PredicateResult(Names):
    """
    Predicate results.

    @see: L{LogLevelFilterPredicate}

    @cvar yes: Log the specified event.  When this value is used,
        L{FilteringLogObserver} will always log the message, without
        evaluating other predicates.

    @cvar no: Do not log the specified event.  When this value is used,
        L{FilteringLogObserver} will I{not} log the message, without
        evaluating other predicates.

    @cvar maybe: Do not have an opinion on the event.  When this value is used,
        L{FilteringLogObserver} will consider subsequent predicate results;
        if returned by the last predicate being considered, then the event will
        be logged.
    """
    yes = NamedConstant()
    no = NamedConstant()
    maybe = NamedConstant()