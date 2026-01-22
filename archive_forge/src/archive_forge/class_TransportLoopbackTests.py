import binascii
import re
import string
import struct
import types
from hashlib import md5, sha1, sha256, sha384, sha512
from typing import Dict, List, Optional, Tuple, Type
from twisted import __version__ as twisted_version
from twisted.conch.error import ConchError
from twisted.conch.ssh import _kex, address, service
from twisted.internet import defer
from twisted.protocols import loopback
from twisted.python import randbytes
from twisted.python.compat import iterbytes
from twisted.python.randbytes import insecureRandom
from twisted.python.reflect import requireModule
from twisted.test import proto_helpers
from twisted.trial.unittest import TestCase
class TransportLoopbackTests(TestCase):
    """
    Test the server transport and client transport against each other,
    """
    if dependencySkip:
        skip = dependencySkip

    def _runClientServer(self, mod):
        """
        Run an async client and server, modifying each using the mod function
        provided.  Returns a Deferred called back when both Protocols have
        disconnected.

        @type mod: C{func}
        @rtype: C{defer.Deferred}
        """
        factory = MockFactory()
        server = transport.SSHServerTransport()
        server.factory = factory
        factory.startFactory()
        server.errors = []
        server.receiveError = lambda code, desc: server.errors.append((code, desc))
        client = transport.SSHClientTransport()
        client.verifyHostKey = lambda x, y: defer.succeed(None)
        client.errors = []
        client.receiveError = lambda code, desc: client.errors.append((code, desc))
        client.connectionSecure = lambda: client.loseConnection()
        server.supportedPublicKeys = list(server.factory.getPublicKeys().keys())
        server = mod(server)
        client = mod(client)

        def check(ignored, server, client):
            name = repr([server.supportedCiphers[0], server.supportedMACs[0], server.supportedKeyExchanges[0], server.supportedCompressions[0]])
            self.assertEqual(client.errors, [])
            self.assertEqual(server.errors, [(transport.DISCONNECT_CONNECTION_LOST, b'user closed connection')])
            if server.supportedCiphers[0] == b'none':
                self.assertFalse(server.isEncrypted(), name)
                self.assertFalse(client.isEncrypted(), name)
            else:
                self.assertTrue(server.isEncrypted(), name)
                self.assertTrue(client.isEncrypted(), name)
            if server.supportedMACs[0] == b'none':
                self.assertFalse(server.isVerified(), name)
                self.assertFalse(client.isVerified(), name)
            else:
                self.assertTrue(server.isVerified(), name)
                self.assertTrue(client.isVerified(), name)
        d = loopback.loopbackAsync(server, client)
        d.addCallback(check, server, client)
        return d

    def test_ciphers(self):
        """
        Test that the client and server play nicely together, in all
        the various combinations of ciphers.
        """
        deferreds = []
        for cipher in transport.SSHTransportBase.supportedCiphers + [b'none']:

            def setCipher(proto):
                proto.supportedCiphers = [cipher]
                return proto
            deferreds.append(self._runClientServer(setCipher))
        return defer.DeferredList(deferreds, fireOnOneErrback=True)

    def test_macs(self):
        """
        Like test_ciphers, but for the various MACs.
        """
        deferreds = []
        for mac in transport.SSHTransportBase.supportedMACs + [b'none']:

            def setMAC(proto):
                proto.supportedMACs = [mac]
                return proto
            deferreds.append(self._runClientServer(setMAC))
        return defer.DeferredList(deferreds, fireOnOneErrback=True)

    def test_keyexchanges(self):
        """
        Like test_ciphers, but for the various key exchanges.
        """
        deferreds = []
        for kexAlgorithm in transport.SSHTransportBase.supportedKeyExchanges:

            def setKeyExchange(proto):
                proto.supportedKeyExchanges = [kexAlgorithm]
                return proto
            deferreds.append(self._runClientServer(setKeyExchange))
        return defer.DeferredList(deferreds, fireOnOneErrback=True)

    def test_compressions(self):
        """
        Like test_ciphers, but for the various compressions.
        """
        deferreds = []
        for compression in transport.SSHTransportBase.supportedCompressions:

            def setCompression(proto):
                proto.supportedCompressions = [compression]
                return proto
            deferreds.append(self._runClientServer(setCompression))
        return defer.DeferredList(deferreds, fireOnOneErrback=True)