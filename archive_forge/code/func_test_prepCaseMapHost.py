from twisted.trial import unittest
from twisted.words.protocols.jabber import jid
def test_prepCaseMapHost(self) -> None:
    """
        Test case mapping of the host part of the JID.
        """
    self.assertEqual(jid.prep('user', 'hoST', 'resource'), ('user', 'host', 'resource'))