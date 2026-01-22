from zope.interface import implementer
from zope.interface.verify import verifyObject
from twisted.conch import telnet
from twisted.internet import defer
from twisted.python.compat import iterbytes
from twisted.test import proto_helpers
from twisted.trial import unittest
def _simpleCommandTest(self, cmdName):
    h = self.p.protocol
    cmd = telnet.IAC + getattr(telnet, cmdName)
    L = [b"Here's some bytes, tra la la", b'But ono!' + cmd + b' an interrupt']
    for b in L:
        self.p.dataReceived(b)
    self.assertEqual(h.calls, [cmdName])
    self.assertEqual(h.data, b''.join(L).replace(cmd, b''))