from unittest import TestCase
from fastimport.helpers import (
from fastimport import (
class TestFileRenameDisplay(TestCase):

    def test_filerename(self):
        c = commands.FileRenameCommand(b'foo/bar', b'foo/baz')
        self.assertEqual(b'R foo/bar foo/baz', bytes(c))

    def test_filerename_quoted(self):
        c = commands.FileRenameCommand(b'foo/b a r', b'foo/b a z')
        self.assertEqual(b'R "foo/b a r" foo/b a z', bytes(c))