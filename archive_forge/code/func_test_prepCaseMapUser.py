from twisted.trial import unittest
from twisted.words.protocols.jabber import jid
def test_prepCaseMapUser(self) -> None:
    """
        Test case mapping of the user part of the JID.
        """
    self.assertEqual(jid.prep('UsEr', 'host', 'resource'), ('user', 'host', 'resource'))