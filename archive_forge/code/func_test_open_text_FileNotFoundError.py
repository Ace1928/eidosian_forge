import unittest
import importlib_resources as resources
from . import data01
from . import util
def test_open_text_FileNotFoundError(self):
    target = resources.files(self.data) / 'does-not-exist'
    with self.assertRaises(FileNotFoundError):
        target.open(encoding='utf-8')