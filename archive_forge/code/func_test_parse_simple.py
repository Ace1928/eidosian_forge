import os
import sys
from .. import osutils, urlutils
from ..errors import PathNotChild
from . import TestCase, TestCaseInTempDir, TestSkipped, features
def test_parse_simple(self):
    parsed = urlutils.URL.from_string('http://example.com:80/one')
    self.assertEqual('http', parsed.scheme)
    self.assertIs(None, parsed.user)
    self.assertIs(None, parsed.password)
    self.assertEqual('example.com', parsed.host)
    self.assertEqual(80, parsed.port)
    self.assertEqual('/one', parsed.path)