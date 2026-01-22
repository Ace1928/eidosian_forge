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
class SSHFactoryTests(unittest.TestCase):
    if not cryptography:
        skip = "can't run without cryptography"

    def makeSSHFactory(self, primes=None):
        sshFactory = factory.SSHFactory()
        sshFactory.getPrimes = lambda: primes
        sshFactory.getPublicKeys = lambda: {b'ssh-rsa': keys.Key.fromString(publicRSA_openssh)}
        sshFactory.getPrivateKeys = lambda: {b'ssh-rsa': keys.Key.fromString(privateRSA_openssh)}
        sshFactory.startFactory()
        return sshFactory

    def test_buildProtocol(self):
        """
        By default, buildProtocol() constructs an instance of
        SSHServerTransport.
        """
        factory = self.makeSSHFactory()
        protocol = factory.buildProtocol(None)
        self.assertIsInstance(protocol, transport.SSHServerTransport)

    def test_buildProtocolRespectsProtocol(self):
        """
        buildProtocol() calls 'self.protocol()' to construct a protocol
        instance.
        """
        calls = []

        def makeProtocol(*args):
            calls.append(args)
            return transport.SSHServerTransport()
        factory = self.makeSSHFactory()
        factory.protocol = makeProtocol
        factory.buildProtocol(None)
        self.assertEqual([()], calls)

    def test_buildProtocolSignatureAlgorithms(self):
        """
        buildProtocol() sets supportedPublicKeys to the list of supported
        signature algorithms.
        """
        f = factory.SSHFactory()
        f.getPublicKeys = lambda: {b'ssh-rsa': keys.Key.fromString(publicRSA_openssh), b'ssh-dss': keys.Key.fromString(publicDSA_openssh)}
        f.getPrivateKeys = lambda: {b'ssh-rsa': keys.Key.fromString(privateRSA_openssh), b'ssh-dss': keys.Key.fromString(privateDSA_openssh)}
        f.startFactory()
        p = f.buildProtocol(None)
        self.assertEqual([b'rsa-sha2-512', b'rsa-sha2-256', b'ssh-rsa', b'ssh-dss'], p.supportedPublicKeys)

    def test_buildProtocolNoPrimes(self):
        """
        Group key exchanges are not supported when we don't have the primes
        database.
        """
        f1 = self.makeSSHFactory(primes=None)
        p1 = f1.buildProtocol(None)
        self.assertNotIn(b'diffie-hellman-group-exchange-sha1', p1.supportedKeyExchanges)
        self.assertNotIn(b'diffie-hellman-group-exchange-sha256', p1.supportedKeyExchanges)

    def test_buildProtocolWithPrimes(self):
        """
        Group key exchanges are supported when we have the primes database.
        """
        f2 = self.makeSSHFactory(primes={1: (2, 3)})
        p2 = f2.buildProtocol(None)
        self.assertIn(b'diffie-hellman-group-exchange-sha1', p2.supportedKeyExchanges)
        self.assertIn(b'diffie-hellman-group-exchange-sha256', p2.supportedKeyExchanges)

    def test_buildProtocolKexECDSA(self):
        """
        ECDSA key exchanges are listed with 256 having a higher priority among ECDSA.
        """
        f2 = self.makeSSHFactory()
        p2 = f2.buildProtocol(None)
        self.assertIn(b'ecdh-sha2-nistp256,ecdh-sha2-nistp384,ecdh-sha2-nistp521', b','.join(p2.supportedKeyExchanges))