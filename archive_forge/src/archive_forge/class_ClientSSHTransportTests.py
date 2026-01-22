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
class ClientSSHTransportTests(ClientSSHTransportBaseCase, TransportTestCase):
    """
    Tests for SSHClientTransport.
    """

    def test_KEXINITMultipleAlgorithms(self):
        """
        Receiving a KEXINIT packet listing multiple supported
        algorithms will set up the first common algorithm, ordered after our
        preference.
        """
        self.proto.dataReceived(b'SSH-2.0-Twisted\r\n\x00\x00\x01\xf4\x04\x14\x99\x99\x99\x99\x99\x99\x99\x99\x99\x99\x99\x99\x99\x99\x99\x99\x00\x00\x00bdiffie-hellman-group1-sha1,diffie-hellman-group-exchange-sha1,diffie-hellman-group-exchange-sha256\x00\x00\x00\x0fssh-dss,ssh-rsa\x00\x00\x00\x85aes128-ctr,aes128-cbc,aes192-ctr,aes192-cbc,aes256-ctr,aes256-cbc,cast128-ctr,cast128-cbc,blowfish-ctr,blowfish-cbc,3des-ctr,3des-cbc\x00\x00\x00\x85aes128-ctr,aes128-cbc,aes192-ctr,aes192-cbc,aes256-ctr,aes256-cbc,cast128-ctr,cast128-cbc,blowfish-ctr,blowfish-cbc,3des-ctr,3des-cbc\x00\x00\x00\x12hmac-md5,hmac-sha1\x00\x00\x00\x12hmac-md5,hmac-sha1\x00\x00\x00\tzlib,none\x00\x00\x00\tzlib,none\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x99\x99\x99\x99')
        self.assertEqual(self.proto.kexAlg, b'diffie-hellman-group-exchange-sha256')
        self.assertEqual(self.proto.keyAlg, b'ssh-rsa')
        self.assertEqual(self.proto.outgoingCompressionType, b'none')
        self.assertEqual(self.proto.incomingCompressionType, b'none')
        ne = self.proto.nextEncryptions
        self.assertEqual(ne.outCipType, b'aes256-ctr')
        self.assertEqual(ne.inCipType, b'aes256-ctr')
        self.assertEqual(ne.outMACType, b'hmac-sha1')
        self.assertEqual(ne.inMACType, b'hmac-sha1')

    def test_notImplementedClientMethods(self):
        """
        verifyHostKey() should return a Deferred which fails with a
        NotImplementedError exception.  connectionSecure() should raise
        NotImplementedError().
        """
        self.assertRaises(NotImplementedError, self.klass().connectionSecure)

        def _checkRaises(f):
            f.trap(NotImplementedError)
        d = self.klass().verifyHostKey(None, None)
        return d.addCallback(self.fail).addErrback(_checkRaises)

    def assertKexInitResponseForDH(self, kexAlgorithm, bits):
        """
        Test that a KEXINIT packet with a group1 or group14 key exchange
        results in a correct KEXDH_INIT response.

        @param kexAlgorithm: The key exchange algorithm to use
        @type kexAlgorithm: L{str}
        """
        self.proto.supportedKeyExchanges = [kexAlgorithm]
        self.proto.dataReceived(self.transport.value())
        x = self.proto.dhSecretKey.private_numbers().x
        self.assertEqual(common.MP(x)[5:], b'\x99' * (bits // 8))
        self.assertEqual(self.packets, [(transport.MSG_KEXDH_INIT, self.proto.dhSecretKeyPublicMP)])

    def test_KEXINIT_group14(self):
        """
        KEXINIT messages requesting diffie-hellman-group14-sha1 result in
        KEXDH_INIT responses.
        """
        self.assertKexInitResponseForDH(b'diffie-hellman-group14-sha1', 2048)

    def test_KEXINIT_badKexAlg(self):
        """
        Test that the client raises a ConchError if it receives a
        KEXINIT message but doesn't have a key exchange algorithm that we
        understand.
        """
        self.proto.supportedKeyExchanges = [b'diffie-hellman-group24-sha1']
        data = self.transport.value().replace(b'group14', b'group24')
        self.assertRaises(ConchError, self.proto.dataReceived, data)

    def test_KEXINITExtensionNegotiation(self):
        """
        If the server sends "ext-info-s" in its key exchange algorithms,
        then the client notes that the server supports extension
        negotiation.  See RFC 8308, section 2.1.
        """
        kexInitPacket = b'\x00' * 16 + common.NS(b'diffie-hellman-group-exchange-sha256,ext-info-s') + common.NS(b'ssh-rsa') + common.NS(b'aes256-ctr') + common.NS(b'aes256-ctr') + common.NS(b'hmac-sha1') + common.NS(b'hmac-sha1') + common.NS(b'none') + common.NS(b'none') + common.NS(b'') + common.NS(b'') + b'\x00' + b'\x00\x00\x00\x00'
        self.proto.ssh_KEXINIT(kexInitPacket)
        self.assertTrue(self.proto._peerSupportsExtensions)

    def begin_KEXDH_REPLY(self):
        """
        Utility for test_KEXDH_REPLY and
        test_disconnectKEXDH_REPLYBadSignature.

        Begins a Diffie-Hellman key exchange in the named group
        Group-14 and computes information needed to return either a
        correct or incorrect signature.

        """
        self.test_KEXINIT_group14()
        f = 2
        fMP = common.MP(f)
        x = self.proto.dhSecretKey.private_numbers().x
        p = self.proto.p
        sharedSecret = _MPpow(f, x, p)
        h = sha1()
        h.update(common.NS(self.proto.ourVersionString) * 2)
        h.update(common.NS(self.proto.ourKexInitPayload) * 2)
        h.update(common.NS(self.blob))
        h.update(self.proto.dhSecretKeyPublicMP)
        h.update(fMP)
        h.update(sharedSecret)
        exchangeHash = h.digest()
        signature = self.privObj.sign(exchangeHash)
        return (exchangeHash, signature, common.NS(self.blob) + fMP)

    def test_KEXDH_REPLY(self):
        """
        Test that the KEXDH_REPLY message verifies the server.
        """
        exchangeHash, signature, packetStart = self.begin_KEXDH_REPLY()

        def _cbTestKEXDH_REPLY(value):
            self.assertIsNone(value)
            self.assertTrue(self.calledVerifyHostKey)
            self.assertEqual(self.proto.sessionID, exchangeHash)
        d = self.proto.ssh_KEX_DH_GEX_GROUP(packetStart + common.NS(signature))
        d.addCallback(_cbTestKEXDH_REPLY)
        return d

    def test_keySetup(self):
        """
        Test that _keySetup sets up the next encryption keys.
        """
        self.proto.kexAlg = b'diffie-hellman-group14-sha1'
        self.proto.nextEncryptions = MockCipher()
        self.simulateKeyExchange(b'AB', b'CD')
        self.assertEqual(self.proto.sessionID, b'CD')
        self.simulateKeyExchange(b'AB', b'EF')
        self.assertEqual(self.proto.sessionID, b'CD')
        self.assertEqual(self.packets[-1], (transport.MSG_NEWKEYS, b''))
        newKeys = [self.proto._getKey(c, b'AB', b'EF') for c in iterbytes(b'ABCDEF')]
        self.assertEqual(self.proto.nextEncryptions.keys, (newKeys[0], newKeys[2], newKeys[1], newKeys[3], newKeys[4], newKeys[5]))

    def test_NEWKEYS(self):
        """
        Test that NEWKEYS transitions the keys from nextEncryptions to
        currentEncryptions.
        """
        self.test_KEXINITMultipleAlgorithms()
        secure = [False]

        def stubConnectionSecure():
            secure[0] = True
        self.proto.connectionSecure = stubConnectionSecure
        self.proto.nextEncryptions = transport.SSHCiphers(b'none', b'none', b'none', b'none')
        self.simulateKeyExchange(b'AB', b'CD')
        self.assertIsNot(self.proto.currentEncryptions, self.proto.nextEncryptions)
        self.proto.nextEncryptions = MockCipher()
        self.proto.ssh_NEWKEYS(b'')
        self.assertIsNone(self.proto.outgoingCompression)
        self.assertIsNone(self.proto.incomingCompression)
        self.assertIs(self.proto.currentEncryptions, self.proto.nextEncryptions)
        self.assertTrue(secure[0])
        self.proto.outgoingCompressionType = b'zlib'
        self.simulateKeyExchange(b'AB', b'GH')
        self.proto.ssh_NEWKEYS(b'')
        self.assertIsNotNone(self.proto.outgoingCompression)
        self.proto.incomingCompressionType = b'zlib'
        self.simulateKeyExchange(b'AB', b'IJ')
        self.proto.ssh_NEWKEYS(b'')
        self.assertIsNotNone(self.proto.incomingCompression)

    def test_SERVICE_ACCEPT(self):
        """
        Test that the SERVICE_ACCEPT packet starts the requested service.
        """
        self.proto.instance = MockService()
        self.proto.ssh_SERVICE_ACCEPT(b'\x00\x00\x00\x0bMockService')
        self.assertTrue(self.proto.instance.started)

    def test_requestService(self):
        """
        Test that requesting a service sends a SERVICE_REQUEST packet.
        """
        self.proto.requestService(MockService())
        self.assertEqual(self.packets, [(transport.MSG_SERVICE_REQUEST, b'\x00\x00\x00\x0bMockService')])

    def test_disconnectKEXDH_REPLYBadSignature(self):
        """
        Test that KEXDH_REPLY disconnects if the signature is bad.
        """
        exchangeHash, signature, packetStart = self.begin_KEXDH_REPLY()
        d = self.proto.ssh_KEX_DH_GEX_GROUP(packetStart + common.NS(b'bad signature'))
        return d.addCallback(lambda _: self.checkDisconnected(transport.DISCONNECT_KEY_EXCHANGE_FAILED))

    def test_disconnectKEX_ECDH_REPLYBadSignature(self):
        """
        Test that KEX_ECDH_REPLY disconnects if the signature is bad.
        """
        kexmsg = b'\xaa' * 16 + common.NS(b'ecdh-sha2-nistp256') + common.NS(b'ssh-rsa') + common.NS(b'aes256-ctr') + common.NS(b'aes256-ctr') + common.NS(b'hmac-sha1') + common.NS(b'hmac-sha1') + common.NS(b'none') + common.NS(b'none') + common.NS(b'') + common.NS(b'') + b'\x00' + b'\x00\x00\x00\x00'
        self.proto.ssh_KEXINIT(kexmsg)
        self.proto.dataReceived(b'SSH-2.0-OpenSSH\r\n')
        self.proto.ecPriv = ec.generate_private_key(ec.SECP256R1(), default_backend())
        self.proto.ecPub = self.proto.ecPriv.public_key()
        thisPriv = ec.generate_private_key(ec.SECP256R1(), default_backend())
        thisPub = thisPriv.public_key()
        encPub = thisPub.public_bytes(serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint)
        self.proto.curve = ec.SECP256R1()
        self.proto.kexAlg = b'ecdh-sha2-nistp256'
        self.proto._ssh_KEX_ECDH_REPLY(common.NS(MockFactory().getPublicKeys()[b'ssh-rsa'].blob()) + common.NS(encPub) + common.NS(b'bad-signature'))
        self.checkDisconnected(transport.DISCONNECT_KEY_EXCHANGE_FAILED)

    def test_disconnectNEWKEYSData(self):
        """
        Test that NEWKEYS disconnects if it receives data.
        """
        self.proto.ssh_NEWKEYS(b'bad packet')
        self.checkDisconnected()

    def test_disconnectSERVICE_ACCEPT(self):
        """
        Test that SERVICE_ACCEPT disconnects if the accepted protocol is
        differet from the asked-for protocol.
        """
        self.proto.instance = MockService()
        self.proto.ssh_SERVICE_ACCEPT(b'\x00\x00\x00\x03bad')
        self.checkDisconnected()

    def test_noPayloadSERVICE_ACCEPT(self):
        """
        Some commercial SSH servers don't send a payload with the
        SERVICE_ACCEPT message.  Conch pretends that it got the correct
        name of the service.
        """
        self.proto.instance = MockService()
        self.proto.ssh_SERVICE_ACCEPT(b'')
        self.assertTrue(self.proto.instance.started)
        self.assertEqual(len(self.packets), 0)