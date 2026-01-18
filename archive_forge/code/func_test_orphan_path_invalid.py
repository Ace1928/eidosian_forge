import io
import unittest
import importlib_resources as resources
from importlib_resources._adapters import (
from . import util
def test_orphan_path_invalid(self):
    with self.assertRaises(ValueError):
        CompatibilityFiles.OrphanPath()