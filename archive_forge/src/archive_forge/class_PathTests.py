import io
import pathlib
import unittest
import importlib_resources as resources
from . import data01
from . import util
class PathTests:

    def test_reading(self):
        """
        Path should be readable and a pathlib.Path instance.
        """
        target = resources.files(self.data) / 'utf-8.file'
        with resources.as_file(target) as path:
            self.assertIsInstance(path, pathlib.Path)
            self.assertTrue(path.name.endswith('utf-8.file'), repr(path))
            self.assertEqual('Hello, UTF-8 world!\n', path.read_text(encoding='utf-8'))