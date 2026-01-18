from twisted.internet import protocol
from twisted.python import failure
from twisted.trial import unittest
from twisted.words.xish import domish, utility, xmlstream
def test_receiveBadXML(self):
    """
        Receiving malformed XML results in an L{STREAM_ERROR_EVENT}.
        """
    streamError = []
    streamEnd = []

    def streamErrorEvent(reason):
        streamError.append(reason)

    def streamEndEvent(_):
        streamEnd.append(None)
    self.xmlstream.addObserver(xmlstream.STREAM_ERROR_EVENT, streamErrorEvent)
    self.xmlstream.addObserver(xmlstream.STREAM_END_EVENT, streamEndEvent)
    self.xmlstream.connectionMade()
    self.xmlstream.dataReceived('<root>')
    self.assertEqual(0, len(streamError))
    self.assertEqual(0, len(streamEnd))
    self.xmlstream.dataReceived('<child><unclosed></child>')
    self.assertEqual(1, len(streamError))
    self.assertTrue(streamError[0].check(domish.ParserError))
    self.assertEqual(1, len(streamEnd))