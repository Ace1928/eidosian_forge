import unittest
import importlib_resources as resources
from . import data01
from . import util
from importlib import import_module
class ReadNamespaceTests(ReadTests, unittest.TestCase):

    def setUp(self):
        from . import namespacedata01
        self.data = namespacedata01