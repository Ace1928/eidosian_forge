from twisted.application import internet
from twisted.trial import unittest
from twisted.words import xmpproutertap as tap
from twisted.words.protocols.jabber import component
def test_portDefault(self) -> None:
    """
        The port option has '5347' as default value
        """
    opt = tap.Options()
    opt.parseOptions([])
    self.assertEqual(opt['port'], 'tcp:5347:interface=127.0.0.1')