from neutron_lib.plugins import directory
from neutron_lib.tests import _base as base
class PluginDirectoryTestCase(base.BaseTestCase):

    def setUp(self):
        super(PluginDirectoryTestCase, self).setUp()
        self.plugin_directory = directory._PluginDirectory()

    def test_add_plugin(self):
        self.plugin_directory.add_plugin('foo', 'bar')
        self.assertEqual(1, len(self.plugin_directory._plugins))

    def test_get_plugin_not_found(self):
        self.assertIsNone(self.plugin_directory.get_plugin('foo'))

    def test_get_plugin_found(self):
        self.plugin_directory._plugins = {'foo': lambda *x, **y: 'bar'}
        plugin = self.plugin_directory.get_plugin('foo')
        self.assertEqual('bar', plugin())

    def test_plugins(self):
        self.plugin_directory._plugins = {'foo': lambda *x, **y: 'bar'}
        self.assertIsNotNone(self.plugin_directory.plugins)

    def test_unique_plugins(self):
        self.plugin_directory._plugins = {'foo1': fake_plugin, 'foo2': fake_plugin}
        self.assertEqual(1, len(self.plugin_directory.unique_plugins))

    def test_is_loaded(self):
        self.assertFalse(self.plugin_directory.is_loaded)
        self.plugin_directory._plugins = {'foo': lambda *x, **y: 'bar'}
        self.assertTrue(self.plugin_directory.is_loaded)