import io
import pathlib
import unittest
import importlib_resources as resources
from . import data01
from . import util
class CommonTests(util.CommonTests, unittest.TestCase):

    def execute(self, package, path):
        with resources.as_file(resources.files(package).joinpath(path)):
            pass