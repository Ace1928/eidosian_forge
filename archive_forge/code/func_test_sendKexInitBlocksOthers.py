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
def test_sendKexInitBlocksOthers(self):
    """
        After L{SSHTransportBase.sendKexInit} has been called, messages types
        other than the following are queued and not sent until after I{NEWKEYS}
        is sent by L{SSHTransportBase._keySetup}.

        RFC 4253, section 7.1.
        """
    disallowedMessageTypes = [transport.MSG_SERVICE_REQUEST, transport.MSG_KEXINIT]
    self.transport.clear()
    del self.proto.sendPacket
    for messageType in disallowedMessageTypes:
        self.proto.sendPacket(messageType, b'foo')
        self.assertEqual(self.transport.value(), b'')
    self.finishKeyExchange(self.proto)
    self.proto.nextEncryptions = MockCipher()
    self.proto._newKeys()
    self.assertEqual(self.transport.value().count(b'foo'), 2)