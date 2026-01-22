from zope.interface import implementer
from zope.interface.verify import verifyObject
from twisted.conch import telnet
from twisted.internet import defer
from twisted.python.compat import iterbytes
from twisted.test import proto_helpers
from twisted.trial import unittest
def _enabledHelper(self, o, eL=[], eR=[], dL=[], dR=[]):
    self.assertEqual(o.enabledLocal, eL)
    self.assertEqual(o.enabledRemote, eR)
    self.assertEqual(o.disabledLocal, dL)
    self.assertEqual(o.disabledRemote, dR)