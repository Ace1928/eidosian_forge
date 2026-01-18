from twisted.trial import unittest
from twisted.words.protocols.jabber import jid
def test_prepNoCaseMapResource(self) -> None:
    """
        Test no case mapping of the resourcce part of the JID.
        """
    self.assertEqual(jid.prep('user', 'hoST', 'resource'), ('user', 'host', 'resource'))
    self.assertNotEqual(jid.prep('user', 'host', 'Resource'), ('user', 'host', 'resource'))