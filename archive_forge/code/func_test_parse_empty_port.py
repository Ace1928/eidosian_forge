import os
import sys
from .. import osutils, urlutils
from ..errors import PathNotChild
from . import TestCase, TestCaseInTempDir, TestSkipped, features
def test_parse_empty_port(self):
    parsed = urlutils.URL.from_string('http://example.com:/one')
    self.assertEqual('http', parsed.scheme)
    self.assertIs(None, parsed.user)
    self.assertIs(None, parsed.password)
    self.assertEqual('example.com', parsed.host)
    self.assertIs(None, parsed.port)
    self.assertEqual('/one', parsed.path)