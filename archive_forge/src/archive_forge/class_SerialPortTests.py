from twisted.internet.error import ConnectionDone
from twisted.internet.protocol import Protocol
from twisted.python.failure import Failure
from twisted.trial import unittest
class SerialPortTests(unittest.TestCase):
    """
    Minimal testing for Twisted's serial port support.

    See ticket #2462 for the eventual full test suite.
    """
    if serialport is None:
        skip = 'Serial port support is not available.'

    def test_connectionMadeLost(self):
        """
        C{connectionMade} and C{connectionLost} are called on the protocol by
        the C{SerialPort}.
        """

        class DummySerialPort(serialport.SerialPort):
            _serialFactory = DoNothing

            def _finishPortSetup(self):
                pass
        events = []

        class SerialProtocol(Protocol):

            def connectionMade(self):
                events.append('connectionMade')

            def connectionLost(self, reason):
                events.append(('connectionLost', reason))
        port = DummySerialPort(SerialProtocol(), '', reactor=DoNothing())
        self.assertEqual(events, ['connectionMade'])
        f = Failure(ConnectionDone())
        port.connectionLost(f)
        self.assertEqual(events, ['connectionMade', ('connectionLost', f)])