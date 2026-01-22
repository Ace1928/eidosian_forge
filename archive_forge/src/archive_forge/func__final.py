from zope.interface import implementer
from zope.interface.verify import verifyObject
from twisted.conch import telnet
from twisted.internet import defer
from twisted.python.compat import iterbytes
from twisted.test import proto_helpers
from twisted.trial import unittest
def _final(x):
    self.p.dataReceived(telnet.IAC + telnet.WONT + b'$')
    self._enabledHelper(self.p.protocol, dR=[b'$'])
    self.p.protocol.remoteEnableable = (b'$',)
    d = self.p.do(b'$')
    self.p.dataReceived(telnet.IAC + telnet.WILL + b'$')
    d.addCallback(self.assertEqual, True)
    d.addCallback(lambda _: self._enabledHelper(self.p.protocol, eR=[b'$'], dR=[b'$']))
    return d