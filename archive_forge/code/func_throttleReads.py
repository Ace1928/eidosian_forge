import sys
from typing import Optional, Type
from zope.interface import directlyProvides, providedBy
from twisted.internet import error, interfaces
from twisted.internet.interfaces import ILoggingContext
from twisted.internet.protocol import ClientFactory, Protocol, ServerFactory
from twisted.python import log
def throttleReads(self):
    """
        Throttle reads on all protocols.
        """
    log.msg('Throttling reads on %s' % self)
    for p in self.protocols.keys():
        p.throttleReads()