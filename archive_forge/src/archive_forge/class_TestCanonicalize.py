import linecache
import os
import re
import sys
from .. import errors, lazy_import, osutils
from . import TestCase, TestCaseInTempDir
import %(root_name)s.%(sub_name)s.%(submoda_name)s as submoda7
class TestCanonicalize(TestCase):
    """Test that we can canonicalize import texts"""

    def check(self, expected, text):
        proc = lazy_import.ImportProcessor()
        parsed = proc._canonicalize_import_text(text)
        self.assertEqual(expected, parsed, 'Incorrect parsing of text:\n%s\n%s\n!=\n%s' % (text, expected, parsed))

    def test_import_one(self):
        self.check(['import one'], 'import one')
        self.check(['import one'], '\nimport one\n\n')

    def test_import_one_two(self):
        self.check(['import one, two'], 'import one, two')
        self.check(['import one, two'], '\nimport one, two\n\n')

    def test_import_one_as_two_as(self):
        self.check(['import one as x, two as y'], 'import one as x, two as y')
        self.check(['import one as x, two as y'], '\nimport one as x, two as y\n')

    def test_from_one_import_two(self):
        self.check(['from one import two'], 'from one import two')
        self.check(['from one import two'], '\nfrom one import two\n\n')
        self.check(['from one import two'], '\nfrom one import (two)\n')
        self.check(['from one import  two '], '\nfrom one import (\n\ttwo\n)\n')

    def test_multiple(self):
        self.check(['import one', 'import two, three', 'from one import four'], 'import one\nimport two, three\nfrom one import four')
        self.check(['import one', 'import two, three', 'from one import four'], 'import one\nimport (two, three)\nfrom one import four')
        self.check(['import one', 'import two, three', 'from one import four'], 'import one\nimport two, three\nfrom one import four')
        self.check(['import one', 'import two, three', 'from one import  four, '], 'import one\nimport two, three\nfrom one import (\n    four,\n    )\n')

    def test_missing_trailing(self):
        proc = lazy_import.ImportProcessor()
        self.assertRaises(lazy_import.InvalidImportLine, proc._canonicalize_import_text, 'from foo import (\n  bar\n')