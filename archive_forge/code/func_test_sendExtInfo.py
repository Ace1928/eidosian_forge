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
def test_sendExtInfo(self):
    """
        Test that EXT_INFO messages are sent correctly.  See RFC 8308,
        section 2.3.
        """
    self.proto._peerSupportsExtensions = True
    self.proto.sendExtInfo([(b'server-sig-algs', b'ssh-rsa,rsa-sha2-256'), (b'elevation', b'd')])
    self.assertEqual(self.packets, [(transport.MSG_EXT_INFO, b'\x00\x00\x00\x02' + common.NS(b'server-sig-algs') + common.NS(b'ssh-rsa,rsa-sha2-256') + common.NS(b'elevation') + common.NS(b'd'))])