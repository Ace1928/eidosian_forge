import io
import unittest
import importlib_resources as resources
from importlib_resources._adapters import (
from . import util
def test_wrap_spec(self):
    spec = wrap_spec(self.package)
    self.assertIsInstance(spec.loader.get_resource_reader(None), CompatibilityFiles)