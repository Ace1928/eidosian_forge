from twisted.trial import unittest
from twisted.words.protocols.jabber import jid
def test_noneArguments(self) -> None:
    """
        Test that using no arguments raises an exception.
        """
    self.assertRaises(RuntimeError, jid.JID)