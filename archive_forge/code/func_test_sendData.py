import struct
from twisted.conch.ssh import channel
from twisted.conch.test import test_userauth
from twisted.python.reflect import requireModule
from twisted.trial import unittest
from twisted.conch import error
def test_sendData(self):
    """
        Test that channel data messages are sent in the right format.
        """
    channel = TestChannel()
    self._openChannel(channel)
    self.conn.sendData(channel, b'a')
    channel.localClosed = True
    self.conn.sendData(channel, b'b')
    self.assertEqual(self.transport.packets, [(connection.MSG_CHANNEL_DATA, b'\x00\x00\x00\xff' + common.NS(b'a'))])