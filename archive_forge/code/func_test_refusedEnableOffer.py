from zope.interface import implementer
from zope.interface.verify import verifyObject
from twisted.conch import telnet
from twisted.internet import defer
from twisted.python.compat import iterbytes
from twisted.test import proto_helpers
from twisted.trial import unittest
def test_refusedEnableOffer(self):
    """
        If the peer refuses to allow us to enable an option, the L{Deferred}
        returned by L{TelnetProtocol.will} fires with an L{OptionRefused}
        L{Failure}.
        """
    self.p.protocol.localEnableable = (b'B',)
    d = self.p.will(b'B')
    self.assertEqual(self.t.value(), telnet.IAC + telnet.WILL + b'B')
    s = self.p.getOptionState(b'B')
    self.assertEqual(s.him.state, 'no')
    self.assertEqual(s.us.state, 'no')
    self.assertFalse(s.him.negotiating)
    self.assertTrue(s.us.negotiating)
    self.p.dataReceived(telnet.IAC + telnet.DONT + b'B')
    d = self.assertFailure(d, telnet.OptionRefused)
    d.addCallback(lambda ignored: self._enabledHelper(self.p.protocol))
    d.addCallback(lambda ignored: self.assertFalse(s.us.negotiating))
    return d