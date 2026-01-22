import base64
import inspect
import re
from io import BytesIO
from typing import Any, List, Optional, Tuple, Type
from zope.interface import directlyProvides, implementer
import twisted.cred.checkers
import twisted.cred.credentials
import twisted.cred.error
import twisted.cred.portal
from twisted import cred
from twisted.cred.checkers import AllowAnonymousAccess, ICredentialsChecker
from twisted.cred.credentials import IAnonymous
from twisted.cred.error import UnauthorizedLogin
from twisted.cred.portal import IRealm, Portal
from twisted.internet import address, defer, error, interfaces, protocol, reactor, task
from twisted.internet.testing import MemoryReactor, StringTransport
from twisted.mail import smtp
from twisted.mail._cred import LOGINCredentials
from twisted.protocols import basic, loopback
from twisted.python.util import LineLog
from twisted.trial.unittest import TestCase
class FakeSMTPServer(basic.LineReceiver):
    clientData = [b'220 hello', b'250 nice to meet you', b'250 great', b'250 great', b'354 go on, lad']

    def connectionMade(self):
        self.buffer = []
        self.clientData = self.clientData[:]
        self.clientData.reverse()
        self.sendLine(self.clientData.pop())

    def lineReceived(self, line):
        self.buffer.append(line)
        if line == b'QUIT':
            self.transport.write(b'221 see ya around\r\n')
            self.transport.loseConnection()
        elif line == b'.':
            self.transport.write(b'250 gotcha\r\n')
        elif line == b'RSET':
            self.transport.loseConnection()
        if self.clientData:
            self.sendLine(self.clientData.pop())