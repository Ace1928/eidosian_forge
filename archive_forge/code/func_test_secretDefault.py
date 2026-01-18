from twisted.application import internet
from twisted.trial import unittest
from twisted.words import xmpproutertap as tap
from twisted.words.protocols.jabber import component
def test_secretDefault(self) -> None:
    """
        The secret option has 'secret' as default value
        """
    opt = tap.Options()
    opt.parseOptions([])
    self.assertEqual(opt['secret'], 'secret')