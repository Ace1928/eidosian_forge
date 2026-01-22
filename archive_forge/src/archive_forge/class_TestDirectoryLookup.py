from .. import transport, urlutils
from ..directory_service import (AliasDirectory, DirectoryServiceRegistry,
from . import TestCase, TestCaseWithTransport
class TestDirectoryLookup(TestCase):

    def setUp(self):
        super().setUp()
        self.registry = DirectoryServiceRegistry()
        self.registry.register('foo:', FooService, 'Map foo URLs to http urls')

    def test_get_directory_service(self):
        directory, suffix = self.registry.get_prefix('foo:bar')
        self.assertIs(FooService, directory)
        self.assertEqual('bar', suffix)

    def test_dereference(self):
        self.assertEqual(FooService.base + 'bar', self.registry.dereference('foo:bar'))
        self.assertEqual(FooService.base + 'bar', self.registry.dereference('foo:bar', purpose='write'))
        self.assertEqual('baz:qux', self.registry.dereference('baz:qux'))
        self.assertEqual('baz:qux', self.registry.dereference('baz:qux', purpose='write'))

    def test_get_transport(self):
        directories.register('foo:', FooService, 'Map foo URLs to http urls')
        self.addCleanup(directories.remove, 'foo:')
        self.assertEqual(FooService.base + 'bar/', transport.get_transport('foo:bar').base)