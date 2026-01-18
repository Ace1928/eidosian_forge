import itertools
import logging; log = logging.getLogger(__name__)
from passlib.tests.utils import TestCase
from passlib.pwd import genword, default_charsets
from passlib.pwd import genphrase
def test_self_info_rate(self):
    """_self_info_rate()"""
    from passlib.pwd import _self_info_rate
    self.assertEqual(_self_info_rate(''), 0)
    self.assertEqual(_self_info_rate('a' * 8), 0)
    self.assertEqual(_self_info_rate('ab'), 1)
    self.assertEqual(_self_info_rate('ab' * 8), 1)
    self.assertEqual(_self_info_rate('abcd'), 2)
    self.assertEqual(_self_info_rate('abcd' * 8), 2)
    self.assertAlmostEqual(_self_info_rate('abcdaaaa'), 1.5488, places=4)