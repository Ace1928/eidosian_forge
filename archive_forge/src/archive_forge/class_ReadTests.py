import unittest
import importlib_resources as resources
from . import data01
from . import util
from importlib import import_module
class ReadTests:

    def test_read_bytes(self):
        result = resources.files(self.data).joinpath('binary.file').read_bytes()
        self.assertEqual(result, bytes(range(4)))

    def test_read_text_default_encoding(self):
        result = resources.files(self.data).joinpath('utf-8.file').read_text(encoding='utf-8')
        self.assertEqual(result, 'Hello, UTF-8 world!\n')

    def test_read_text_given_encoding(self):
        result = resources.files(self.data).joinpath('utf-16.file').read_text(encoding='utf-16')
        self.assertEqual(result, 'Hello, UTF-16 world!\n')

    def test_read_text_with_errors(self):
        """
        Raises UnicodeError without the 'errors' argument.
        """
        target = resources.files(self.data) / 'utf-16.file'
        self.assertRaises(UnicodeError, target.read_text, encoding='utf-8')
        result = target.read_text(encoding='utf-8', errors='ignore')
        self.assertEqual(result, 'H\x00e\x00l\x00l\x00o\x00,\x00 \x00U\x00T\x00F\x00-\x001\x006\x00 \x00w\x00o\x00r\x00l\x00d\x00!\x00\n\x00')