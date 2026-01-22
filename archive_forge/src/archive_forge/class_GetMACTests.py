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
class GetMACTests(TestCase):
    """
    Tests for L{SSHCiphers._getMAC}.
    """
    if dependencySkip:
        skip = dependencySkip

    def setUp(self):
        self.ciphers = transport.SSHCiphers(b'A', b'B', b'C', b'D')

    def getSharedSecret(self):
        """
        Generate a new shared secret to be used with the tests.

        @return: A new secret.
        @rtype: L{bytes}
        """
        return insecureRandom(64)

    def assertGetMAC(self, hmacName, hashProcessor, digestSize, blockPadSize):
        """
        Check that when L{SSHCiphers._getMAC} is called with a supportd HMAC
        algorithm name it returns a tuple of
        (digest object, inner pad, outer pad, digest size) with a C{key}
        attribute set to the value of the key supplied.

        @param hmacName: Identifier of HMAC algorithm.
        @type hmacName: L{bytes}

        @param hashProcessor: Callable for the hash algorithm.
        @type hashProcessor: C{callable}

        @param digestSize: Size of the digest for algorithm.
        @type digestSize: L{int}

        @param blockPadSize: Size of padding applied to the shared secret to
            match the block size.
        @type blockPadSize: L{int}
        """
        secret = self.getSharedSecret()
        params = self.ciphers._getMAC(hmacName, secret)
        key = secret[:digestSize] + b'\x00' * blockPadSize
        innerPad = bytes((ord(b) ^ 54 for b in iterbytes(key)))
        outerPad = bytes((ord(b) ^ 92 for b in iterbytes(key)))
        self.assertEqual((hashProcessor, innerPad, outerPad, digestSize), params)
        self.assertEqual(key, params.key)

    def test_hmacsha2512(self):
        """
        When L{SSHCiphers._getMAC} is called with the C{b"hmac-sha2-512"} MAC
        algorithm name it returns a tuple of (sha512 digest object, inner pad,
        outer pad, sha512 digest size) with a C{key} attribute set to the
        value of the key supplied.
        """
        self.assertGetMAC(b'hmac-sha2-512', sha512, digestSize=64, blockPadSize=64)

    def test_hmacsha2384(self):
        """
        When L{SSHCiphers._getMAC} is called with the C{b"hmac-sha2-384"} MAC
        algorithm name it returns a tuple of (sha384 digest object, inner pad,
        outer pad, sha384 digest size) with a C{key} attribute set to the
        value of the key supplied.
        """
        self.assertGetMAC(b'hmac-sha2-384', sha384, digestSize=48, blockPadSize=80)

    def test_hmacsha2256(self):
        """
        When L{SSHCiphers._getMAC} is called with the C{b"hmac-sha2-256"} MAC
        algorithm name it returns a tuple of (sha256 digest object, inner pad,
        outer pad, sha256 digest size) with a C{key} attribute set to the
        value of the key supplied.
        """
        self.assertGetMAC(b'hmac-sha2-256', sha256, digestSize=32, blockPadSize=32)

    def test_hmacsha1(self):
        """
        When L{SSHCiphers._getMAC} is called with the C{b"hmac-sha1"} MAC
        algorithm name it returns a tuple of (sha1 digest object, inner pad,
        outer pad, sha1 digest size) with a C{key} attribute set to the value
        of the key supplied.
        """
        self.assertGetMAC(b'hmac-sha1', sha1, digestSize=20, blockPadSize=44)

    def test_hmacmd5(self):
        """
        When L{SSHCiphers._getMAC} is called with the C{b"hmac-md5"} MAC
        algorithm name it returns a tuple of (md5 digest object, inner pad,
        outer pad, md5 digest size) with a C{key} attribute set to the value of
        the key supplied.
        """
        self.assertGetMAC(b'hmac-md5', md5, digestSize=16, blockPadSize=48)

    def test_none(self):
        """
        When L{SSHCiphers._getMAC} is called with the C{b"none"} MAC algorithm
        name it returns a tuple of (None, "", "", 0).
        """
        key = self.getSharedSecret()
        params = self.ciphers._getMAC(b'none', key)
        self.assertEqual((None, b'', b'', 0), params)