from twisted.trial import unittest
from twisted.words.protocols.jabber.xmpp_stringprep import (
class DeprecationTests(unittest.TestCase):
    """
    Deprecations in L{twisted.words.protocols.jabber.xmpp_stringprep}.
    """

    def test_crippled(self) -> None:
        """
        L{xmpp_stringprep.crippled} is deprecated and always returns C{False}.
        """
        from twisted.words.protocols.jabber.xmpp_stringprep import crippled
        warnings = self.flushWarnings(offendingFunctions=[self.test_crippled])
        self.assertEqual(DeprecationWarning, warnings[0]['category'])
        self.assertEqual('twisted.words.protocols.jabber.xmpp_stringprep.crippled was deprecated in Twisted 13.1.0: crippled is always False', warnings[0]['message'])
        self.assertEqual(1, len(warnings))
        self.assertEqual(crippled, False)