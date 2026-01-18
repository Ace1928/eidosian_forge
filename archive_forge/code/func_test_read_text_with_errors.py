import unittest
import importlib_resources as resources
from . import data01
from . import util
from importlib import import_module
def test_read_text_with_errors(self):
    """
        Raises UnicodeError without the 'errors' argument.
        """
    target = resources.files(self.data) / 'utf-16.file'
    self.assertRaises(UnicodeError, target.read_text, encoding='utf-8')
    result = target.read_text(encoding='utf-8', errors='ignore')
    self.assertEqual(result, 'H\x00e\x00l\x00l\x00o\x00,\x00 \x00U\x00T\x00F\x00-\x001\x006\x00 \x00w\x00o\x00r\x00l\x00d\x00!\x00\n\x00')