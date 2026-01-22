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
class ClientSSHTransportDHGroupExchangeBaseCase(ClientSSHTransportBaseCase):
    """
    Diffie-Hellman group exchange tests for SSHClientTransport.
    """
    '\n    1536-bit modulus from RFC 3526\n    '
    P1536 = int('FFFFFFFFFFFFFFFFC90FDAA22168C234C4C6628B80DC1CD129024E088A67CC74020BBEA63B139B22514A08798E3404DDEF9519B3CD3A431B302B0A6DF25F14374FE1356D6D51C245E485B576625E7EC6F44C42E9A637ED6B0BFF5CB6F406B7EDEE386BFB5A899FA5AE9F24117C4B1FE649286651ECE45B3DC2007CB8A163BF0598DA48361C55D39A69163FA8FD24CF5F83655D23DCA3AD961C62F356208552BB9ED529077096966D670C354E4ABC9804F1746C08CA237327FFFFFFFFFFFFFFFF', 16)

    def test_KEXINIT_groupexchange(self):
        """
        KEXINIT packet with a group-exchange key exchange results
        in a KEX_DH_GEX_REQUEST message.
        """
        self.proto.supportedKeyExchanges = [self.kexAlgorithm]
        self.proto.dataReceived(self.transport.value())
        self.assertEqual(self.packets, [(transport.MSG_KEX_DH_GEX_REQUEST, b'\x00\x00\x04\x00\x00\x00\x08\x00\x00\x00 \x00')])

    def test_KEX_DH_GEX_GROUP(self):
        """
        Test that the KEX_DH_GEX_GROUP message results in a
        KEX_DH_GEX_INIT message with the client's Diffie-Hellman public key.
        """
        self.test_KEXINIT_groupexchange()
        self.proto.ssh_KEX_DH_GEX_GROUP(common.MP(self.P1536) + common.MP(2))
        self.assertEqual(self.proto.p, self.P1536)
        self.assertEqual(self.proto.g, 2)
        x = self.proto.dhSecretKey.private_numbers().x
        self.assertEqual(common.MP(x)[5:], b'\x99' * 192)
        self.assertEqual(self.proto.dhSecretKeyPublicMP, common.MP(pow(2, x, self.P1536)))
        self.assertEqual(self.packets[1:], [(transport.MSG_KEX_DH_GEX_INIT, self.proto.dhSecretKeyPublicMP)])

    def begin_KEX_DH_GEX_REPLY(self):
        """
        Utility for test_KEX_DH_GEX_REPLY and
        test_disconnectGEX_REPLYBadSignature.

        Begins a Diffie-Hellman key exchange in an unnamed
        (server-specified) group and computes information needed to
        return either a correct or incorrect signature.
        """
        self.test_KEX_DH_GEX_GROUP()
        p = self.proto.p
        f = 3
        fMP = common.MP(f)
        sharedSecret = _MPpow(f, self.proto.dhSecretKey.private_numbers().x, p)
        h = self.hashProcessor()
        h.update(common.NS(self.proto.ourVersionString) * 2)
        h.update(common.NS(self.proto.ourKexInitPayload) * 2)
        h.update(common.NS(self.blob))
        h.update(b'\x00\x00\x04\x00\x00\x00\x08\x00\x00\x00 \x00')
        h.update(common.MP(self.P1536) + common.MP(2))
        h.update(self.proto.dhSecretKeyPublicMP)
        h.update(fMP)
        h.update(sharedSecret)
        exchangeHash = h.digest()
        signature = self.privObj.sign(exchangeHash)
        return (exchangeHash, signature, common.NS(self.blob) + fMP)

    def test_KEX_DH_GEX_REPLY(self):
        """
        Test that the KEX_DH_GEX_REPLY message results in a verified
        server.
        """
        exchangeHash, signature, packetStart = self.begin_KEX_DH_GEX_REPLY()

        def _cbTestKEX_DH_GEX_REPLY(value):
            self.assertIsNone(value)
            self.assertTrue(self.calledVerifyHostKey)
            self.assertEqual(self.proto.sessionID, exchangeHash)
        d = self.proto.ssh_KEX_DH_GEX_REPLY(packetStart + common.NS(signature))
        d.addCallback(_cbTestKEX_DH_GEX_REPLY)
        return d

    def test_disconnectGEX_REPLYBadSignature(self):
        """
        Test that KEX_DH_GEX_REPLY disconnects if the signature is bad.
        """
        exchangeHash, signature, packetStart = self.begin_KEX_DH_GEX_REPLY()
        d = self.proto.ssh_KEX_DH_GEX_REPLY(packetStart + common.NS(b'bad signature'))
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