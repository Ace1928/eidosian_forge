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