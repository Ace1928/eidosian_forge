import struct
from itertools import chain
from typing import Dict, List, Tuple
from twisted.conch.test.keydata import (
from twisted.conch.test.loopback import LoopbackRelay
from twisted.cred import portal
from twisted.cred.error import UnauthorizedLogin
from twisted.internet import defer, protocol, reactor
from twisted.internet.error import ProcessTerminated
from twisted.python import failure, log
from twisted.python.reflect import requireModule
from twisted.trial import unittest
from twisted.python import components
def test_notEnoughData(self):
    """
        When the string passed to L{common.getMP} doesn't even make 5 bytes,
        it should raise a L{struct.error}.
        """
    self.assertRaises(struct.error, self.getMP, b'\x02\x00')