import sys
import unittest
import importlib_resources as resources
import pathlib
from . import data01
from . import util
from importlib import import_module
class ResourceCornerCaseTests(unittest.TestCase):

    def test_package_has_no_reader_fallback(self):
        """
        Test odd ball packages which:
        # 1. Do not have a ResourceReader as a loader
        # 2. Are not on the file system
        # 3. Are not in a zip file
        """
        module = util.create_package(file=data01, path=data01.__file__, contents=['A', 'B', 'C'])
        module.__loader__ = object()
        module.__file__ = '/path/which/shall/not/be/named'
        module.__spec__.loader = module.__loader__
        module.__spec__.origin = module.__file__
        self.assertFalse(resources.files(module).joinpath('A').is_file())