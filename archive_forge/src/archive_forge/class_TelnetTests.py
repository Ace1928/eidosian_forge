from zope.interface import implementer
from zope.interface.verify import verifyObject
from twisted.conch import telnet
from twisted.internet import defer
from twisted.python.compat import iterbytes
from twisted.test import proto_helpers
from twisted.trial import unittest
class TelnetTests(unittest.TestCase):
    """
    Tests for L{telnet.Telnet}.

    L{telnet.Telnet} implements the TELNET protocol (RFC 854), including option
    and suboption negotiation, and option state tracking.
    """

    def setUp(self):
        """
        Create an unconnected L{telnet.Telnet} to be used by tests.
        """
        self.protocol = TestTelnet()

    def test_enableLocal(self):
        """
        L{telnet.Telnet.enableLocal} should reject all options, since
        L{telnet.Telnet} does not know how to implement any options.
        """
        self.assertFalse(self.protocol.enableLocal(b'\x00'))

    def test_enableRemote(self):
        """
        L{telnet.Telnet.enableRemote} should reject all options, since
        L{telnet.Telnet} does not know how to implement any options.
        """
        self.assertFalse(self.protocol.enableRemote(b'\x00'))

    def test_disableLocal(self):
        """
        It is an error for L{telnet.Telnet.disableLocal} to be called, since
        L{telnet.Telnet.enableLocal} will never allow any options to be enabled
        locally.  If a subclass overrides enableLocal, it must also override
        disableLocal.
        """
        self.assertRaises(NotImplementedError, self.protocol.disableLocal, b'\x00')

    def test_disableRemote(self):
        """
        It is an error for L{telnet.Telnet.disableRemote} to be called, since
        L{telnet.Telnet.enableRemote} will never allow any options to be
        enabled remotely.  If a subclass overrides enableRemote, it must also
        override disableRemote.
        """
        self.assertRaises(NotImplementedError, self.protocol.disableRemote, b'\x00')

    def test_requestNegotiation(self):
        """
        L{telnet.Telnet.requestNegotiation} formats the feature byte and the
        payload bytes into the subnegotiation format and sends them.

        See RFC 855.
        """
        transport = proto_helpers.StringTransport()
        self.protocol.makeConnection(transport)
        self.protocol.requestNegotiation(b'\x01', b'\x02\x03')
        self.assertEqual(transport.value(), b'\xff\xfa\x01\x02\x03\xff\xf0')

    def test_requestNegotiationEscapesIAC(self):
        """
        If the payload for a subnegotiation includes I{IAC}, it is escaped by
        L{telnet.Telnet.requestNegotiation} with another I{IAC}.

        See RFC 855.
        """
        transport = proto_helpers.StringTransport()
        self.protocol.makeConnection(transport)
        self.protocol.requestNegotiation(b'\x01', b'\xff')
        self.assertEqual(transport.value(), b'\xff\xfa\x01\xff\xff\xff\xf0')

    def _deliver(self, data, *expected):
        """
        Pass the given bytes to the protocol's C{dataReceived} method and
        assert that the given events occur.
        """
        received = self.protocol.events = []
        self.protocol.dataReceived(data)
        self.assertEqual(received, list(expected))

    def test_oneApplicationDataByte(self):
        """
        One application-data byte in the default state gets delivered right
        away.
        """
        self._deliver(b'a', ('bytes', b'a'))

    def test_twoApplicationDataBytes(self):
        """
        Two application-data bytes in the default state get delivered
        together.
        """
        self._deliver(b'bc', ('bytes', b'bc'))

    def test_threeApplicationDataBytes(self):
        """
        Three application-data bytes followed by a control byte get
        delivered, but the control byte doesn't.
        """
        self._deliver(b'def' + telnet.IAC, ('bytes', b'def'))

    def test_escapedControl(self):
        """
        IAC in the escaped state gets delivered and so does another
        application-data byte following it.
        """
        self._deliver(telnet.IAC)
        self._deliver(telnet.IAC + b'g', ('bytes', telnet.IAC + b'g'))

    def test_carriageReturn(self):
        """
        A carriage return only puts the protocol into the newline state.  A
        linefeed in the newline state causes just the newline to be
        delivered.  A nul in the newline state causes a carriage return to
        be delivered.  An IAC in the newline state causes a carriage return
        to be delivered and puts the protocol into the escaped state.
        Anything else causes a carriage return and that thing to be
        delivered.
        """
        self._deliver(b'\r')
        self._deliver(b'\n', ('bytes', b'\n'))
        self._deliver(b'\r\n', ('bytes', b'\n'))
        self._deliver(b'\r')
        self._deliver(b'\x00', ('bytes', b'\r'))
        self._deliver(b'\r\x00', ('bytes', b'\r'))
        self._deliver(b'\r')
        self._deliver(b'a', ('bytes', b'\ra'))
        self._deliver(b'\ra', ('bytes', b'\ra'))
        self._deliver(b'\r')
        self._deliver(telnet.IAC + telnet.IAC + b'x', ('bytes', b'\r' + telnet.IAC + b'x'))

    def test_applicationDataBeforeSimpleCommand(self):
        """
        Application bytes received before a command are delivered before the
        command is processed.
        """
        self._deliver(b'x' + telnet.IAC + telnet.NOP, ('bytes', b'x'), ('command', telnet.NOP, None))

    def test_applicationDataBeforeCommand(self):
        """
        Application bytes received before a WILL/WONT/DO/DONT are delivered
        before the command is processed.
        """
        self.protocol.commandMap = {}
        self._deliver(b'y' + telnet.IAC + telnet.WILL + b'\x00', ('bytes', b'y'), ('command', telnet.WILL, b'\x00'))

    def test_applicationDataBeforeSubnegotiation(self):
        """
        Application bytes received before a subnegotiation command are
        delivered before the negotiation is processed.
        """
        self._deliver(b'z' + telnet.IAC + telnet.SB + b'Qx' + telnet.IAC + telnet.SE, ('bytes', b'z'), ('negotiate', b'Q', [b'x']))