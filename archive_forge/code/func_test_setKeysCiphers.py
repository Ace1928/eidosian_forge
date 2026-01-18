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
def test_setKeysCiphers(self):
    """
        Test that setKeys sets up the ciphers.
        """
    key = b'\x00' * 64
    for cipName in transport.SSHTransportBase.supportedCiphers:
        modName, keySize, counter = transport.SSHCiphers.cipherMap[cipName]
        encCipher = transport.SSHCiphers(cipName, b'none', b'none', b'none')
        decCipher = transport.SSHCiphers(b'none', cipName, b'none', b'none')
        cip = encCipher._getCipher(cipName, key, key)
        bs = cip.algorithm.block_size // 8
        encCipher.setKeys(key, key, b'', b'', b'', b'')
        decCipher.setKeys(b'', b'', key, key, b'', b'')
        self.assertEqual(encCipher.encBlockSize, bs)
        self.assertEqual(decCipher.decBlockSize, bs)
        encryptor = cip.encryptor()
        enc = encryptor.update(key[:bs])
        enc2 = encryptor.update(key[:bs])
        self.assertEqual(encCipher.encrypt(key[:bs]), enc)
        self.assertEqual(encCipher.encrypt(key[:bs]), enc2)
        self.assertEqual(decCipher.decrypt(enc), key[:bs])
        self.assertEqual(decCipher.decrypt(enc2), key[:bs])