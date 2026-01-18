from zope.interface import implementer
from zope.interface.verify import verifyObject
from twisted.conch import telnet
from twisted.internet import defer
from twisted.python.compat import iterbytes
from twisted.test import proto_helpers
from twisted.trial import unittest
def test_refusedEnableRequest(self):
    """
        If the peer refuses to enable an option we request it to enable, the
        L{Deferred} returned by L{TelnetProtocol.do} fires with an
        L{OptionRefused} L{Failure}.
        """
    self.p.protocol.remoteEnableable = (b'B',)
    d = self.p.do(b'B')
    self.assertEqual(self.t.value(), telnet.IAC + telnet.DO + b'B')
    s = self.p.getOptionState(b'B')
    self.assertEqual(s.him.state, 'no')
    self.assertEqual(s.us.state, 'no')
    self.assertTrue(s.him.negotiating)
    self.assertFalse(s.us.negotiating)
    self.p.dataReceived(telnet.IAC + telnet.WONT + b'B')
    d = self.assertFailure(d, telnet.OptionRefused)
    d.addCallback(lambda ignored: self._enabledHelper(self.p.protocol))
    d.addCallback(lambda ignored: self.assertFalse(s.him.negotiating))
    return d