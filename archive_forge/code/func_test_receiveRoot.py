from twisted.internet import protocol
from twisted.python import failure
from twisted.trial import unittest
from twisted.words.xish import domish, utility, xmlstream
def test_receiveRoot(self):
    """
        Receiving the starttag of the root element results in stream start.
        """
    streamStarted = []

    def streamStartEvent(rootelem):
        streamStarted.append(None)
    self.xmlstream.addObserver(xmlstream.STREAM_START_EVENT, streamStartEvent)
    self.xmlstream.connectionMade()
    self.xmlstream.dataReceived('<root>')
    self.assertEqual(1, len(streamStarted))