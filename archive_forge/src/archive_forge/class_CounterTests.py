from Cryptodome.Util.py3compat import *
import unittest
class CounterTests(unittest.TestCase):

    def setUp(self):
        global Counter
        from Cryptodome.Util import Counter

    def test_BE(self):
        """Big endian"""
        c = Counter.new(128)
        c = Counter.new(128, little_endian=False)

    def test_LE(self):
        """Little endian"""
        c = Counter.new(128, little_endian=True)

    def test_nbits(self):
        c = Counter.new(nbits=128)
        self.assertRaises(ValueError, Counter.new, 129)

    def test_prefix(self):
        c = Counter.new(128, prefix=b('xx'))

    def test_suffix(self):
        c = Counter.new(128, suffix=b('xx'))

    def test_iv(self):
        c = Counter.new(128, initial_value=2)
        self.assertRaises(ValueError, Counter.new, 16, initial_value=131071)