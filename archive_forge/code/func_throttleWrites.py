import sys
from typing import Optional, Type
from zope.interface import directlyProvides, providedBy
from twisted.internet import error, interfaces
from twisted.internet.interfaces import ILoggingContext
from twisted.internet.protocol import ClientFactory, Protocol, ServerFactory
from twisted.python import log
def throttleWrites(self):
    """
        Throttle writes on all protocols.
        """
    log.msg('Throttling writes on %s' % self)
    for p in self.protocols.keys():
        p.throttleWrites()