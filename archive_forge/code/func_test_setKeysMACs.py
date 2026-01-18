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
def test_setKeysMACs(self):
    """
        Test that setKeys sets up the MACs.
        """
    key = b'\x00' * 64
    for macName, mod in transport.SSHCiphers.macMap.items():
        outMac = transport.SSHCiphers(b'none', b'none', macName, b'none')
        inMac = transport.SSHCiphers(b'none', b'none', b'none', macName)
        outMac.setKeys(b'', b'', b'', b'', key, b'')
        inMac.setKeys(b'', b'', b'', b'', b'', key)
        if mod:
            ds = mod().digest_size
        else:
            ds = 0
        self.assertEqual(inMac.verifyDigestSize, ds)
        if mod:
            mod, i, o, ds = outMac._getMAC(macName, key)
        seqid = 0
        data = key
        packet = b'\x00' * 4 + key
        if mod:
            mac = mod(o + mod(i + packet).digest()).digest()
        else:
            mac = b''
        self.assertEqual(outMac.makeMAC(seqid, data), mac)
        self.assertTrue(inMac.verify(seqid, data, mac))