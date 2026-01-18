import struct
from twisted.conch.ssh import channel
from twisted.conch.test import test_userauth
from twisted.python.reflect import requireModule
from twisted.trial import unittest
from twisted.conch import error
def test_sendRequest(self):
    """
        Test that channel request messages are sent in the right format.
        """
    channel = TestChannel()
    self._openChannel(channel)
    d = self.conn.sendRequest(channel, b'test', b'test', True)
    d.addErrback(lambda failure: None)
    self.conn.sendRequest(channel, b'test2', b'', False)
    channel.localClosed = True
    self.conn.sendRequest(channel, b'test3', b'', True)
    self.assertEqual(self.transport.packets, [(connection.MSG_CHANNEL_REQUEST, b'\x00\x00\x00\xff' + common.NS(b'test') + b'\x01test'), (connection.MSG_CHANNEL_REQUEST, b'\x00\x00\x00\xff' + common.NS(b'test2') + b'\x00')])
    self.assertEqual(self.conn.deferreds[0], [d])