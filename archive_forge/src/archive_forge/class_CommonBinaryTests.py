import unittest
import importlib_resources as resources
from . import data01
from . import util
from importlib import import_module
class CommonBinaryTests(util.CommonTests, unittest.TestCase):

    def execute(self, package, path):
        resources.files(package).joinpath(path).read_bytes()