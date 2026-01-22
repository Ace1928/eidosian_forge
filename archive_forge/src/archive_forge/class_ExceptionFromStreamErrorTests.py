from twisted.trial import unittest
from twisted.words.protocols.jabber import error
from twisted.words.xish import domish
class ExceptionFromStreamErrorTests(unittest.TestCase):

    def test_basic(self) -> None:
        """
        Test basic operations of exceptionFromStreamError.

        Given a realistic stream error, check if a sane exception is returned.

        Using this error::

          <stream:error xmlns:stream='http://etherx.jabber.org/streams'>
            <xml-not-well-formed xmlns='urn:ietf:params:xml:ns:xmpp-streams'/>
          </stream:error>
        """
        e = domish.Element(('http://etherx.jabber.org/streams', 'error'))
        e.addElement((NS_XMPP_STREAMS, 'xml-not-well-formed'))
        result = error.exceptionFromStreamError(e)
        self.assertIsInstance(result, error.StreamError)
        self.assertEqual('xml-not-well-formed', result.condition)