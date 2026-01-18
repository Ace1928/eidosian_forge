import unittest
from oslo_config import iniparser
def test_parseerror(self):
    exc = iniparser.ParseError('test', 42, 'example')
    self.assertEqual(str(exc), "at line 42, test: 'example'")