from neutron_lib.plugins import directory
from neutron_lib.tests import _base as base
class DirectoryTestCase(base.BaseTestCase):

    def test__create_plugin_directory(self):
        self.assertIsNotNone(directory._create_plugin_directory())

    def test__get_plugin_directory(self):
        self.assertIsNotNone(directory._get_plugin_directory())

    def test_add_plugin(self):
        directory.add_plugin('foo', fake_plugin)
        self.assertIn('foo', directory.get_plugins())

    def test_get_plugin_core_none(self):
        self.assertIsNone(directory.get_plugin())

    def test_get_plugin_alias_none(self):
        self.assertIsNone(directory.get_plugin('foo'))

    def test_get_plugin_core(self):
        directory.add_plugin('CORE', fake_plugin)
        self.assertIsNotNone(directory.get_plugin())

    def test_get_plugin_alias(self):
        directory.add_plugin('foo', fake_plugin)
        self.assertIsNotNone(directory.get_plugin('foo'))

    def test_get_plugins_none(self):
        self.assertFalse(directory.get_plugins())

    def test_get_unique_plugins_none(self):
        self.assertFalse(directory.get_unique_plugins())

    def test_get_plugins(self):
        directory.add_plugin('CORE', fake_plugin)
        self.assertIsNotNone(directory.get_plugins())

    def test_get_unique_plugins(self):
        directory.add_plugin('foo1', fake_plugin)
        directory.add_plugin('foo2', fake_plugin)
        self.assertEqual(1, len(directory.get_unique_plugins()))

    def test_is_loaded(self):
        self.assertFalse(directory.is_loaded())
        directory.add_plugin('foo1', fake_plugin)
        self.assertTrue(directory.is_loaded())