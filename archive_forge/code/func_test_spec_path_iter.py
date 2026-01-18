import io
import unittest
import importlib_resources as resources
from importlib_resources._adapters import (
from . import util
def test_spec_path_iter(self):
    self.assertEqual(sorted((path.name for path in self.files.iterdir())), ['a', 'b', 'c'])