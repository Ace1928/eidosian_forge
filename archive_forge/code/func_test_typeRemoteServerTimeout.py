from twisted.trial import unittest
from twisted.words.protocols.jabber import error
from twisted.words.xish import domish
def test_typeRemoteServerTimeout(self) -> None:
    """
        Remote Server Timeout should yield type wait, code 504.
        """
    e = error.StanzaError('remote-server-timeout')
    self.assertEqual('wait', e.type)
    self.assertEqual('504', e.code)