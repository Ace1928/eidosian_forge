from hashlib import sha1
from zope.interface.verify import verifyObject
from twisted.python import failure
from twisted.trial import unittest
from twisted.words.protocols.jabber import component, ijabber, xmlstream
from twisted.words.protocols.jabber.jid import JID
from twisted.words.xish import domish
from twisted.words.xish.utility import XmlPipe
def test_onError(self):
    """
        An observer for stream errors should trigger onError to log it.
        """
    self.xmlstream.dispatch(self.xmlstream, xmlstream.STREAM_CONNECTED_EVENT)

    class TestError(Exception):
        pass
    reason = failure.Failure(TestError())
    self.xmlstream.dispatch(reason, xmlstream.STREAM_ERROR_EVENT)
    self.assertEqual(1, len(self.flushLoggedErrors(TestError)))