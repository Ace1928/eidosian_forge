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
class ClientSSHTransportBaseCase(ServerAndClientSSHTransportBaseCase):
    """
    Base case for SSHClientTransport tests.
    """
    klass: Optional[Type[transport.SSHTransportBase]] = transport.SSHClientTransport

    def verifyHostKey(self, pubKey, fingerprint):
        """
        Mock version of SSHClientTransport.verifyHostKey.
        """
        self.calledVerifyHostKey = True
        self.assertEqual(pubKey, self.blob)
        self.assertEqual(fingerprint.replace(b':', b''), binascii.hexlify(md5(pubKey).digest()))
        return defer.succeed(True)

    def setUp(self):
        TransportTestCase.setUp(self)
        self.blob = keys.Key.fromString(keydata.publicRSA_openssh).blob()
        self.privObj = keys.Key.fromString(keydata.privateRSA_openssh)
        self.calledVerifyHostKey = False
        self.proto.verifyHostKey = self.verifyHostKey