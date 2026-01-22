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
class ConchTestClientAuth(userauth.SSHUserAuthClient):
    hasTriedNone = 0
    canSucceedPublicKey = 0
    canSucceedPassword = 0

    def ssh_USERAUTH_SUCCESS(self, packet):
        if not self.canSucceedPassword and self.canSucceedPublicKey:
            raise unittest.FailTest('got USERAUTH_SUCCESS before password and publickey')
        userauth.SSHUserAuthClient.ssh_USERAUTH_SUCCESS(self, packet)

    def getPassword(self):
        self.canSucceedPassword = 1
        return defer.succeed(b'testpass')

    def getPrivateKey(self):
        self.canSucceedPublicKey = 1
        return defer.succeed(keys.Key.fromString(privateDSA_openssh))

    def getPublicKey(self):
        return keys.Key.fromString(publicDSA_openssh)