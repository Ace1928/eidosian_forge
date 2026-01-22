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
class ClientSSHTransportECDHBaseCase(ClientSSHTransportBaseCase):
    """
    Elliptic Curve Diffie-Hellman tests for SSHClientTransport.
    """

    def test_KEXINIT(self):
        """
        KEXINIT packet with an elliptic curve key exchange results
        in a KEXDH_INIT message.
        """
        self.proto.supportedKeyExchanges = [self.kexAlgorithm]
        self.proto.dataReceived(self.transport.value())
        self.assertEqual(self.packets, [(transport.MSG_KEXDH_INIT, common.NS(self.proto._encodeECPublicKey(self.proto.ecPub)))])

    def begin_KEXDH_REPLY(self):
        """
        Utility for test_KEXDH_REPLY and
        test_disconnectKEXDH_REPLYBadSignature.

        Begins an Elliptic Curve Diffie-Hellman key exchange and computes
        information needed to return either a correct or incorrect
        signature.
        """
        self.test_KEXINIT()
        privKey = MockFactory().getPrivateKeys()[b'ssh-rsa']
        pubKey = MockFactory().getPublicKeys()[b'ssh-rsa']
        ecPriv = self.proto._generateECPrivateKey()
        ecPub = ecPriv.public_key()
        encPub = self.proto._encodeECPublicKey(ecPub)
        sharedSecret = self.proto._generateECSharedSecret(ecPriv, self.proto._encodeECPublicKey(self.proto.ecPub))
        h = self.hashProcessor()
        h.update(common.NS(self.proto.ourVersionString))
        h.update(common.NS(self.proto.otherVersionString))
        h.update(common.NS(self.proto.ourKexInitPayload))
        h.update(common.NS(self.proto.otherKexInitPayload))
        h.update(common.NS(pubKey.blob()))
        h.update(common.NS(self.proto._encodeECPublicKey(self.proto.ecPub)))
        h.update(common.NS(encPub))
        h.update(sharedSecret)
        exchangeHash = h.digest()
        signature = privKey.sign(exchangeHash)
        return (exchangeHash, signature, common.NS(pubKey.blob()) + common.NS(encPub))

    def test_KEXDH_REPLY(self):
        """
        Test that the KEXDH_REPLY message completes the key exchange.
        """
        exchangeHash, signature, packetStart = self.begin_KEXDH_REPLY()

        def _cbTestKEXDH_REPLY(value):
            self.assertIsNone(value)
            self.assertTrue(self.calledVerifyHostKey)
            self.assertEqual(self.proto.sessionID, exchangeHash)
        d = self.proto.ssh_KEX_DH_GEX_GROUP(packetStart + common.NS(signature))
        d.addCallback(_cbTestKEXDH_REPLY)
        return d

    def test_disconnectKEXDH_REPLYBadSignature(self):
        """
        Test that KEX_ECDH_REPLY disconnects if the signature is bad.
        """
        exchangeHash, signature, packetStart = self.begin_KEXDH_REPLY()
        d = self.proto.ssh_KEX_DH_GEX_GROUP(packetStart + common.NS(b'bad signature'))
        return d.addCallback(lambda _: self.checkDisconnected(transport.DISCONNECT_KEY_EXCHANGE_FAILED))