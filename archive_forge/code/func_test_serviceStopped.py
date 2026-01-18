import struct
from twisted.conch.ssh import channel
from twisted.conch.test import test_userauth
from twisted.python.reflect import requireModule
from twisted.trial import unittest
from twisted.conch import error
def test_serviceStopped(self):
    """
        Test that serviceStopped() closes any open channels.
        """
    channel1 = TestChannel()
    channel2 = TestChannel()
    self.conn.openChannel(channel1)
    self.conn.openChannel(channel2)
    self.conn.ssh_CHANNEL_OPEN_CONFIRMATION(b'\x00\x00\x00\x00' * 4)
    self.assertTrue(channel1.gotOpen)
    self.assertFalse(channel1.gotClosed)
    self.assertFalse(channel2.gotOpen)
    self.assertFalse(channel2.gotClosed)
    self.conn.serviceStopped()
    self.assertTrue(channel1.gotClosed)
    self.assertFalse(channel2.gotOpen)
    self.assertFalse(channel2.gotClosed)
    from twisted.internet.error import ConnectionLost
    self.assertIsInstance(channel2.openFailureReason, ConnectionLost)