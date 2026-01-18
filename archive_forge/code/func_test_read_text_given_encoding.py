import unittest
import importlib_resources as resources
from . import data01
from . import util
from importlib import import_module
def test_read_text_given_encoding(self):
    result = resources.files(self.data).joinpath('utf-16.file').read_text(encoding='utf-16')
    self.assertEqual(result, 'Hello, UTF-16 world!\n')