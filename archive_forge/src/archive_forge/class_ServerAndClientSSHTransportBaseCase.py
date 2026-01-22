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
class ServerAndClientSSHTransportBaseCase:
    """
    Tests that need to be run on both the server and the client.
    """

    def checkDisconnected(self, kind=None):
        """
        Helper function to check if the transport disconnected.
        """
        if kind is None:
            kind = transport.DISCONNECT_PROTOCOL_ERROR
        self.assertEqual(self.packets[-1][0], transport.MSG_DISCONNECT)
        self.assertEqual(self.packets[-1][1][3:4], bytes((kind,)))

    def connectModifiedProtocol(self, protoModification, kind=None):
        """
        Helper function to connect a modified protocol to the test protocol
        and test for disconnection.
        """
        if kind is None:
            kind = transport.DISCONNECT_KEY_EXCHANGE_FAILED
        proto2 = self.klass()
        protoModification(proto2)
        proto2.makeConnection(proto_helpers.StringTransport())
        self.proto.dataReceived(proto2.transport.value())
        if kind:
            self.checkDisconnected(kind)
        return proto2

    def test_disconnectIfCantMatchKex(self):
        """
        Test that the transport disconnects if it can't match the key
        exchange
        """

        def blankKeyExchanges(proto2):
            proto2.supportedKeyExchanges = []
        self.connectModifiedProtocol(blankKeyExchanges)

    def test_disconnectIfCantMatchKeyAlg(self):
        """
        Like test_disconnectIfCantMatchKex, but for the key algorithm.
        """

        def blankPublicKeys(proto2):
            proto2.supportedPublicKeys = []
        self.connectModifiedProtocol(blankPublicKeys)

    def test_disconnectIfCantMatchCompression(self):
        """
        Like test_disconnectIfCantMatchKex, but for the compression.
        """

        def blankCompressions(proto2):
            proto2.supportedCompressions = []
        self.connectModifiedProtocol(blankCompressions)

    def test_disconnectIfCantMatchCipher(self):
        """
        Like test_disconnectIfCantMatchKex, but for the encryption.
        """

        def blankCiphers(proto2):
            proto2.supportedCiphers = []
        self.connectModifiedProtocol(blankCiphers)

    def test_disconnectIfCantMatchMAC(self):
        """
        Like test_disconnectIfCantMatchKex, but for the MAC.
        """

        def blankMACs(proto2):
            proto2.supportedMACs = []
        self.connectModifiedProtocol(blankMACs)

    def test_getPeer(self):
        """
        Test that the transport's L{getPeer} method returns an
        L{SSHTransportAddress} with the L{IAddress} of the peer.
        """
        self.assertEqual(self.proto.getPeer(), address.SSHTransportAddress(self.proto.transport.getPeer()))

    def test_getHost(self):
        """
        Test that the transport's L{getHost} method returns an
        L{SSHTransportAddress} with the L{IAddress} of the host.
        """
        self.assertEqual(self.proto.getHost(), address.SSHTransportAddress(self.proto.transport.getHost()))