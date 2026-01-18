import sys
import unittest
import importlib_resources as resources
import pathlib
from . import data01
from . import util
from importlib import import_module
def test_resource_contents(self):
    package = util.create_package(file=data01, path=data01.__file__, contents=['A', 'B', 'C'])
    self.assertEqual(names(resources.files(package)), {'A', 'B', 'C'})