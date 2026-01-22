from __future__ import unicode_literals
import errno
import posixpath
from unittest import TestCase
from pybtex import io
class IOTest(TestCase):

    def setUp(self):
        self.fs = MockFilesystem(files=('/home/test/foo.bib', '/home/test/foo.bbl', '/usr/share/texmf/bibtex/bst/unsrt.bst'), writable_dirs=('/home/test',), readonly_dirs='/')
        self.fs.chdir('/home/test')

    def test_open_existing(self):
        file = io._open_existing(self.fs.open, 'foo.bbl', 'rb', locate=self.fs.locate)
        self.assertEqual(file.name, '/home/test/foo.bbl')
        self.assertEqual(file.mode, 'rb')

    def test_open_missing(self):
        self.assertRaises(EnvironmentError, io._open_existing, self.fs.open, 'nosuchfile.bbl', 'rb', locate=self.fs.locate)

    def test_locate(self):
        file = io._open_existing(self.fs.open, 'unsrt.bst', 'rb', locate=self.fs.locate)
        self.assertEqual(file.name, '/usr/share/texmf/bibtex/bst/unsrt.bst')
        self.assertEqual(file.mode, 'rb')

    def test_create(self):
        file = io._open_or_create(self.fs.open, 'foo.bbl', 'wb', {})
        self.assertEqual(file.name, '/home/test/foo.bbl')
        self.assertEqual(file.mode, 'wb')

    def test_create_in_readonly_dir(self):
        self.fs.chdir('/')
        self.assertRaises(EnvironmentError, io._open_or_create, self.fs.open, 'foo.bbl', 'wb', {})

    def test_create_in_fallback_dir(self):
        self.fs.chdir('/')
        file = io._open_or_create(self.fs.open, 'foo.bbl', 'wb', {'TEXMFOUTPUT': '/home/test'})
        self.assertEqual(file.name, '/home/test/foo.bbl')
        self.assertEqual(file.mode, 'wb')