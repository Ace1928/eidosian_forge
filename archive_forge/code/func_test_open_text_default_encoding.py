import unittest
import importlib_resources as resources
from . import data01
from . import util
def test_open_text_default_encoding(self):
    target = resources.files(self.data) / 'utf-8.file'
    with target.open(encoding='utf-8') as fp:
        result = fp.read()
        self.assertEqual(result, 'Hello, UTF-8 world!\n')