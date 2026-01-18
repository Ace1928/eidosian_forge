import struct
from twisted.conch.ssh import channel
from twisted.conch.test import test_userauth
from twisted.python.reflect import requireModule
from twisted.trial import unittest
from twisted.conch import error
def test_sendClose(self):
    """
        Test that channel close messages are sent in the right format.
        """
    channel = TestChannel()
    self._openChannel(channel)
    self.conn.sendClose(channel)
    self.assertTrue(channel.localClosed)
    self.assertEqual(self.transport.packets, [(connection.MSG_CHANNEL_CLOSE, b'\x00\x00\x00\xff')])
    self.conn.sendClose(channel)
    self.assertEqual(self.transport.packets, [(connection.MSG_CHANNEL_CLOSE, b'\x00\x00\x00\xff')])
    channel2 = TestChannel()
    self._openChannel(channel2)
    self.assertTrue(channel2.gotOpen)
    self.assertFalse(channel2.gotClosed)
    channel2.remoteClosed = True
    self.conn.sendClose(channel2)
    self.assertTrue(channel2.gotClosed)