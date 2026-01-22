import sys
from typing import Optional, Type
from zope.interface import directlyProvides, providedBy
from twisted.internet import error, interfaces
from twisted.internet.interfaces import ILoggingContext
from twisted.internet.protocol import ClientFactory, Protocol, ServerFactory
from twisted.python import log
class ThrottlingFactory(WrappingFactory):
    """
    Throttles bandwidth and number of connections.

    Write bandwidth will only be throttled if there is a producer
    registered.
    """
    protocol = ThrottlingProtocol

    def __init__(self, wrappedFactory, maxConnectionCount=sys.maxsize, readLimit=None, writeLimit=None):
        WrappingFactory.__init__(self, wrappedFactory)
        self.connectionCount = 0
        self.maxConnectionCount = maxConnectionCount
        self.readLimit = readLimit
        self.writeLimit = writeLimit
        self.readThisSecond = 0
        self.writtenThisSecond = 0
        self.unthrottleReadsID = None
        self.checkReadBandwidthID = None
        self.unthrottleWritesID = None
        self.checkWriteBandwidthID = None

    def callLater(self, period, func):
        """
        Wrapper around
        L{reactor.callLater<twisted.internet.interfaces.IReactorTime.callLater>}
        for test purpose.
        """
        from twisted.internet import reactor
        return reactor.callLater(period, func)

    def registerWritten(self, length):
        """
        Called by protocol to tell us more bytes were written.
        """
        self.writtenThisSecond += length

    def registerRead(self, length):
        """
        Called by protocol to tell us more bytes were read.
        """
        self.readThisSecond += length

    def checkReadBandwidth(self):
        """
        Checks if we've passed bandwidth limits.
        """
        if self.readThisSecond > self.readLimit:
            self.throttleReads()
            throttleTime = float(self.readThisSecond) / self.readLimit - 1.0
            self.unthrottleReadsID = self.callLater(throttleTime, self.unthrottleReads)
        self.readThisSecond = 0
        self.checkReadBandwidthID = self.callLater(1, self.checkReadBandwidth)

    def checkWriteBandwidth(self):
        if self.writtenThisSecond > self.writeLimit:
            self.throttleWrites()
            throttleTime = float(self.writtenThisSecond) / self.writeLimit - 1.0
            self.unthrottleWritesID = self.callLater(throttleTime, self.unthrottleWrites)
        self.writtenThisSecond = 0
        self.checkWriteBandwidthID = self.callLater(1, self.checkWriteBandwidth)

    def throttleReads(self):
        """
        Throttle reads on all protocols.
        """
        log.msg('Throttling reads on %s' % self)
        for p in self.protocols.keys():
            p.throttleReads()

    def unthrottleReads(self):
        """
        Stop throttling reads on all protocols.
        """
        self.unthrottleReadsID = None
        log.msg('Stopped throttling reads on %s' % self)
        for p in self.protocols.keys():
            p.unthrottleReads()

    def throttleWrites(self):
        """
        Throttle writes on all protocols.
        """
        log.msg('Throttling writes on %s' % self)
        for p in self.protocols.keys():
            p.throttleWrites()

    def unthrottleWrites(self):
        """
        Stop throttling writes on all protocols.
        """
        self.unthrottleWritesID = None
        log.msg('Stopped throttling writes on %s' % self)
        for p in self.protocols.keys():
            p.unthrottleWrites()

    def buildProtocol(self, addr):
        if self.connectionCount == 0:
            if self.readLimit is not None:
                self.checkReadBandwidth()
            if self.writeLimit is not None:
                self.checkWriteBandwidth()
        if self.connectionCount < self.maxConnectionCount:
            self.connectionCount += 1
            return WrappingFactory.buildProtocol(self, addr)
        else:
            log.msg('Max connection count reached!')
            return None

    def unregisterProtocol(self, p):
        WrappingFactory.unregisterProtocol(self, p)
        self.connectionCount -= 1
        if self.connectionCount == 0:
            if self.unthrottleReadsID is not None:
                self.unthrottleReadsID.cancel()
            if self.checkReadBandwidthID is not None:
                self.checkReadBandwidthID.cancel()
            if self.unthrottleWritesID is not None:
                self.unthrottleWritesID.cancel()
            if self.checkWriteBandwidthID is not None:
                self.checkWriteBandwidthID.cancel()