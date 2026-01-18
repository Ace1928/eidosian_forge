from __future__ import annotations
from unittest import skipIf
from zope.interface.verify import verifyObject
from twisted.trial.unittest import TestCase
def test_writeExtended(self) -> None:
    """
        Test that writeExtended handles data correctly.  Send extended data
        up to the size of the window, splitting the extended data into packets
        of length remoteMaxPacket.
        """
    cb = [False]

    def stubStopWriting() -> None:
        cb[0] = True
    self.channel.stopWriting = stubStopWriting
    self.channel.writeExtended(1, b'd')
    self.channel.writeExtended(1, b'a')
    self.channel.writeExtended(2, b't')
    self.assertFalse(self.channel.areWriting)
    self.assertTrue(cb[0])
    self.channel.addWindowBytes(20)
    self.channel.writeExtended(2, b'a')
    data = self.conn.extData[self.channel]
    self.assertEqual(data, [(1, b'da'), (2, b't'), (2, b'a')])
    self.assertEqual(self.channel.remoteWindowLeft, 16)
    self.channel.writeExtended(3, b'12345678901')
    self.assertEqual(data, [(1, b'da'), (2, b't'), (2, b'a'), (3, b'1234567890'), (3, b'1')])
    self.assertEqual(self.channel.remoteWindowLeft, 5)
    cb[0] = False
    self.channel.writeExtended(4, b'123456')
    self.assertFalse(self.channel.areWriting)
    self.assertTrue(cb[0])
    self.assertEqual(data, [(1, b'da'), (2, b't'), (2, b'a'), (3, b'1234567890'), (3, b'1'), (4, b'12345')])
    self.assertEqual(self.channel.extBuf, [[4, b'6']])
    self.assertEqual(self.channel.remoteWindowLeft, 0)