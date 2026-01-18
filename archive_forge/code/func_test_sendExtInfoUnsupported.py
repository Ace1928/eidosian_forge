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
def test_sendExtInfoUnsupported(self):
    """
        If the peer has not advertised support for extension negotiation, no
        EXT_INFO message is sent, since RFC 8308 only guarantees that the
        peer will be prepared to accept it if it has advertised support.
        """
    self.proto.sendExtInfo([(b'server-sig-algs', b'ssh-rsa,rsa-sha2-256')])
    self.assertEqual(self.packets, [])