import os.path
import sys
import pathlib
import unittest
from importlib import import_module
from importlib_resources.readers import MultiplexedPath, NamespaceReader
class NamespaceReaderTest(unittest.TestCase):
    site_dir = str(pathlib.Path(__file__).parent)

    @classmethod
    def setUpClass(cls):
        sys.path.append(cls.site_dir)

    @classmethod
    def tearDownClass(cls):
        sys.path.remove(cls.site_dir)

    def test_init_error(self):
        with self.assertRaises(ValueError):
            NamespaceReader(['path1', 'path2'])

    def test_resource_path(self):
        namespacedata01 = import_module('namespacedata01')
        reader = NamespaceReader(namespacedata01.__spec__.submodule_search_locations)
        root = os.path.abspath(os.path.join(__file__, '..', 'namespacedata01'))
        self.assertEqual(reader.resource_path('binary.file'), os.path.join(root, 'binary.file'))
        self.assertEqual(reader.resource_path('imaginary'), os.path.join(root, 'imaginary'))

    def test_files(self):
        namespacedata01 = import_module('namespacedata01')
        reader = NamespaceReader(namespacedata01.__spec__.submodule_search_locations)
        root = os.path.abspath(os.path.join(__file__, '..', 'namespacedata01'))
        self.assertIsInstance(reader.files(), MultiplexedPath)
        self.assertEqual(repr(reader.files()), f"MultiplexedPath('{root}')")