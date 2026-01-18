from zope.interface import implementer
from zope.interface.verify import verifyObject
from twisted.conch import telnet
from twisted.internet import defer
from twisted.python.compat import iterbytes
from twisted.test import proto_helpers
from twisted.trial import unittest
def test_threeApplicationDataBytes(self):
    """
        Three application-data bytes followed by a control byte get
        delivered, but the control byte doesn't.
        """
    self._deliver(b'def' + telnet.IAC, ('bytes', b'def'))