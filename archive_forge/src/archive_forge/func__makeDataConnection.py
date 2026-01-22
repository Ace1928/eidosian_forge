import errno
import getpass
import os
import random
import string
from io import BytesIO
from zope.interface import implementer
from zope.interface.verify import verifyClass
from twisted.cred import checkers, credentials, portal
from twisted.cred.error import UnauthorizedLogin
from twisted.cred.portal import IRealm
from twisted.internet import defer, error, protocol, reactor, task
from twisted.internet.interfaces import IConsumer
from twisted.protocols import basic, ftp, loopback
from twisted.python import failure, filepath, runtime
from twisted.test import proto_helpers
from twisted.trial.unittest import TestCase
def _makeDataConnection(self, ignored=None):
    deferred = defer.Deferred()

    class DataFactory(protocol.ServerFactory):
        protocol = _BufferingProtocol

        def buildProtocol(self, addr):
            p = protocol.ServerFactory.buildProtocol(self, addr)
            reactor.callLater(0, deferred.callback, p)
            return p
    dataPort = reactor.listenTCP(0, DataFactory(), interface='127.0.0.1')
    self.dataPorts.append(dataPort)
    cmd = 'PORT ' + ftp.encodeHostPort('127.0.0.1', dataPort.getHost().port)
    self.client.queueStringCommand(cmd)
    return deferred