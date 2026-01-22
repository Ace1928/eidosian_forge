import struct
from itertools import chain
from typing import Dict, List, Tuple
from twisted.conch.test.keydata import (
from twisted.conch.test.loopback import LoopbackRelay
from twisted.cred import portal
from twisted.cred.error import UnauthorizedLogin
from twisted.internet import defer, protocol, reactor
from twisted.internet.error import ProcessTerminated
from twisted.python import failure, log
from twisted.python.reflect import requireModule
from twisted.trial import unittest
from twisted.python import components
class SSHTestChannel(channel.SSHChannel):

    def __init__(self, name, opened, *args, **kwargs):
        self.name = name
        self._opened = opened
        self.received = []
        self.receivedExt = []
        self.onClose = defer.Deferred()
        channel.SSHChannel.__init__(self, *args, **kwargs)

    def openFailed(self, reason):
        self._opened.errback(reason)

    def channelOpen(self, ignore):
        self._opened.callback(self)

    def dataReceived(self, data):
        self.received.append(data)

    def extReceived(self, dataType, data):
        if dataType == connection.EXTENDED_DATA_STDERR:
            self.receivedExt.append(data)
        else:
            log.msg(f'Unrecognized extended data: {dataType!r}')

    def request_exit_status(self, status):
        [self.status] = struct.unpack('>L', status)

    def eofReceived(self):
        self.eofCalled = True

    def closed(self):
        self.onClose.callback(None)