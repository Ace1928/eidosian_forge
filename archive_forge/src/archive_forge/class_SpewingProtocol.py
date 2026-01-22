import sys
from typing import Optional, Type
from zope.interface import directlyProvides, providedBy
from twisted.internet import error, interfaces
from twisted.internet.interfaces import ILoggingContext
from twisted.internet.protocol import ClientFactory, Protocol, ServerFactory
from twisted.python import log
class SpewingProtocol(ProtocolWrapper):

    def dataReceived(self, data):
        log.msg('Received: %r' % data)
        ProtocolWrapper.dataReceived(self, data)

    def write(self, data):
        log.msg('Sending: %r' % data)
        ProtocolWrapper.write(self, data)