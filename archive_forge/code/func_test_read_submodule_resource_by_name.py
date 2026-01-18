import unittest
import importlib_resources as resources
from . import data01
from . import util
from importlib import import_module
def test_read_submodule_resource_by_name(self):
    result = resources.files('namespacedata01.subdirectory').joinpath('binary.file').read_bytes()
    self.assertEqual(result, bytes(range(12, 16)))