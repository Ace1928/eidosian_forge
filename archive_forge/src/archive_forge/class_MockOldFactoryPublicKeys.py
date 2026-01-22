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
class MockOldFactoryPublicKeys(MockFactory):
    """
    The old SSHFactory returned mappings from key names to strings from
    getPublicKeys().  We return those here for testing.
    """

    def getPublicKeys(self):
        """
        We used to map key types to public key blobs as strings.
        """
        keys = MockFactory.getPublicKeys(self)
        for name, key in keys.items()[:]:
            keys[name] = key.blob()
        return keys