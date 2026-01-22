import linecache
import os
import re
import sys
from .. import errors, lazy_import, osutils
from . import TestCase, TestCaseInTempDir
import %(root_name)s.%(sub_name)s.%(submoda_name)s as submoda7
class TestConvertImportToMap(TestCase):
    """Directly test the conversion from import strings to maps"""

    def check(self, expected, import_strings):
        proc = lazy_import.ImportProcessor()
        for import_str in import_strings:
            proc._convert_import_str(import_str)
        self.assertEqual(expected, proc.imports, 'Import of %r was not converted correctly %s != %s' % (import_strings, expected, proc.imports))

    def test_import_one(self):
        self.check({'one': (['one'], None, {})}, ['import one'])

    def test_import_one_two(self):
        one_two_map = {'one': (['one'], None, {'two': (['one', 'two'], None, {})})}
        self.check(one_two_map, ['import one.two'])
        self.check(one_two_map, ['import one, one.two'])
        self.check(one_two_map, ['import one', 'import one.two'])
        self.check(one_two_map, ['import one.two', 'import one'])

    def test_import_one_two_three(self):
        one_two_three_map = {'one': (['one'], None, {'two': (['one', 'two'], None, {'three': (['one', 'two', 'three'], None, {})})})}
        self.check(one_two_three_map, ['import one.two.three'])
        self.check(one_two_three_map, ['import one, one.two.three'])
        self.check(one_two_three_map, ['import one', 'import one.two.three'])
        self.check(one_two_three_map, ['import one.two.three', 'import one'])

    def test_import_one_as_x(self):
        self.check({'x': (['one'], None, {})}, ['import one as x'])

    def test_import_one_two_as_x(self):
        self.check({'x': (['one', 'two'], None, {})}, ['import one.two as x'])

    def test_import_mixed(self):
        mixed = {'x': (['one', 'two'], None, {}), 'one': (['one'], None, {'two': (['one', 'two'], None, {})})}
        self.check(mixed, ['import one.two as x, one.two'])
        self.check(mixed, ['import one.two as x', 'import one.two'])
        self.check(mixed, ['import one.two', 'import one.two as x'])

    def test_import_with_as(self):
        self.check({'fast': (['fast'], None, {})}, ['import fast'])