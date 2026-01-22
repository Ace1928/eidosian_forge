import unittest
import os
from jsbeautifier.unpackers.myobfuscate import detect, unpack
from jsbeautifier.unpackers.tests import __path__ as path
class TestMyObfuscate(unittest.TestCase):
    """MyObfuscate obfuscator testcase."""

    @classmethod
    def setUpClass(cls):
        """Load source files (encoded and decoded version) for tests."""
        with open(INPUT, 'r') as data:
            cls.input = data.read()
        with open(OUTPUT, 'r') as data:
            cls.output = data.read()

    def test_detect(self):
        """Test detect() function."""

        def detected(source):
            return self.assertTrue(detect(source))
        detected(self.input)

    def test_unpack(self):
        """Test unpack() function."""

        def check(inp, out):
            return self.assertEqual(unpack(inp), out)
        check(self.input, self.output)