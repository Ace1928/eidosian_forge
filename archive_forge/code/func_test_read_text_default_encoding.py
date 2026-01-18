import unittest
import importlib_resources as resources
from . import data01
from . import util
from importlib import import_module
def test_read_text_default_encoding(self):
    result = resources.files(self.data).joinpath('utf-8.file').read_text(encoding='utf-8')
    self.assertEqual(result, 'Hello, UTF-8 world!\n')