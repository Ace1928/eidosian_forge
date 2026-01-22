import sys
from typing import Optional, Type
from zope.interface import directlyProvides, providedBy
from twisted.internet import error, interfaces
from twisted.internet.interfaces import ILoggingContext
from twisted.internet.protocol import ClientFactory, Protocol, ServerFactory
from twisted.python import log
class TimeoutFactory(WrappingFactory):
    """
    Factory for TimeoutWrapper.
    """
    protocol = TimeoutProtocol

    def __init__(self, wrappedFactory, timeoutPeriod=30 * 60):
        self.timeoutPeriod = timeoutPeriod
        WrappingFactory.__init__(self, wrappedFactory)

    def buildProtocol(self, addr):
        return self.protocol(self, self.wrappedFactory.buildProtocol(addr), timeoutPeriod=self.timeoutPeriod)

    def callLater(self, period, func):
        """
        Wrapper around
        L{reactor.callLater<twisted.internet.interfaces.IReactorTime.callLater>}
        for test purpose.
        """
        from twisted.internet import reactor
        return reactor.callLater(period, func)