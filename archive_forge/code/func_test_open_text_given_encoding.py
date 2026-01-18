import unittest
import importlib_resources as resources
from . import data01
from . import util
def test_open_text_given_encoding(self):
    target = resources.files(self.data) / 'utf-16.file'
    with target.open(encoding='utf-16', errors='strict') as fp:
        result = fp.read()
    self.assertEqual(result, 'Hello, UTF-16 world!\n')