from twisted.internet import protocol
from twisted.python import failure
from twisted.trial import unittest
from twisted.words.xish import domish, utility, xmlstream
def test_streamEnd(self):
    """
        Ending the stream fires a L{STREAM_END_EVENT}.
        """
    streamEnd = []

    def streamEndEvent(reason):
        streamEnd.append(reason)
    self.xmlstream.addObserver(xmlstream.STREAM_END_EVENT, streamEndEvent)
    self.xmlstream.connectionMade()
    self.loseConnection()
    self.assertEqual(1, len(streamEnd))
    self.assertIsInstance(streamEnd[0], failure.Failure)
    self.assertEqual(streamEnd[0].getErrorMessage(), self.connectionLostMsg)