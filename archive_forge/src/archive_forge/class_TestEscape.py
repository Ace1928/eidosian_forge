import re
import sys
import textwrap
from io import StringIO
from .. import commands, export_pot, option, registry, tests
class TestEscape(tests.TestCase):

    def test_simple_escape(self):
        self.assertEqual(export_pot._escape('foobar'), 'foobar')
        s = 'foo\nbar\r\tbaz\\"spam"'
        e = 'foo\\nbar\\r\\tbaz\\\\\\"spam\\"'
        self.assertEqual(export_pot._escape(s), e)

    def test_complex_escape(self):
        s = '\\r \\\n'
        e = '\\\\r \\\\\\n'
        self.assertEqual(export_pot._escape(s), e)