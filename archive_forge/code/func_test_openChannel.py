import struct
from twisted.conch.ssh import channel
from twisted.conch.test import test_userauth
from twisted.python.reflect import requireModule
from twisted.trial import unittest
from twisted.conch import error
def test_openChannel(self):
    """
        Test that open channel messages are sent in the right format.
        """
    channel = TestChannel()
    self.conn.openChannel(channel, b'aaaa')
    self.assertEqual(self.transport.packets, [(connection.MSG_CHANNEL_OPEN, common.NS(b'TestChannel') + b'\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x80\x00aaaa')])
    self.assertEqual(channel.id, 0)
    self.assertEqual(self.conn.localChannelID, 1)