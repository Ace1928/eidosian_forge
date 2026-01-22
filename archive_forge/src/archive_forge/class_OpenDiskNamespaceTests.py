import unittest
import importlib_resources as resources
from . import data01
from . import util
class OpenDiskNamespaceTests(OpenTests, unittest.TestCase):

    def setUp(self):
        from . import namespacedata01
        self.data = namespacedata01