import sys
import types
from heat.engine import plugin_manager
from heat.tests import common
class TestPluginManager(common.HeatTestCase):

    @staticmethod
    def module():
        return sys.modules[__name__]

    def test_load_single_mapping(self):
        pm = plugin_manager.PluginMapping('current_test')
        self.assertEqual(current_test_mapping(), pm.load_from_module(self.module()))

    def test_load_first_alternative_mapping(self):
        pm = plugin_manager.PluginMapping(['current_test', 'legacy_test'])
        self.assertEqual(current_test_mapping(), pm.load_from_module(self.module()))

    def test_load_second_alternative_mapping(self):
        pm = plugin_manager.PluginMapping(['nonexist', 'current_test'])
        self.assertEqual(current_test_mapping(), pm.load_from_module(self.module()))

    def test_load_mapping_args(self):
        pm = plugin_manager.PluginMapping('args_test', 'baz', 'quux')
        expected = {0: 'baz', 1: 'quux'}
        self.assertEqual(expected, pm.load_from_module(self.module()))

    def test_load_mapping_kwargs(self):
        pm = plugin_manager.PluginMapping('kwargs_test', baz='quux')
        self.assertEqual({'baz': 'quux'}, pm.load_from_module(self.module()))

    def test_load_mapping_non_existent(self):
        pm = plugin_manager.PluginMapping('nonexist')
        self.assertEqual({}, pm.load_from_module(self.module()))

    def test_load_mapping_error(self):
        pm = plugin_manager.PluginMapping('error_test')
        self.assertRaises(MappingTestError, pm.load_from_module, self.module())

    def test_load_mapping_exception(self):
        pm = plugin_manager.PluginMapping('error_test_exception')
        self.assertRaisesRegex(Exception, 'exception', pm.load_from_module, self.module())

    def test_load_mapping_invalidtype(self):
        pm = plugin_manager.PluginMapping('invalid_type_test')
        self.assertEqual({}, pm.load_from_module(self.module()))

    def test_load_mapping_nonereturn(self):
        pm = plugin_manager.PluginMapping('none_return_test')
        self.assertEqual({}, pm.load_from_module(self.module()))

    def test_modules(self):
        mgr = plugin_manager.PluginManager('heat.tests')
        for module in mgr.modules:
            self.assertEqual(types.ModuleType, type(module))
            self.assertTrue(module.__name__.startswith('heat.tests') or module.__name__.startswith('heat.engine.plugins'))

    def test_load_all_skip_tests(self):
        mgr = plugin_manager.PluginManager('heat.tests')
        pm = plugin_manager.PluginMapping('current_test')
        all_items = pm.load_all(mgr)
        for item in current_test_mapping().items():
            self.assertNotIn(item, all_items)

    def test_load_all(self):
        import heat.tests.engine.test_plugin_manager
        mgr = plugin_manager.PluginManager('heat.tests')
        pm = plugin_manager.PluginMapping('current_test')
        mgr.modules = [heat.tests.engine.test_plugin_manager]
        all_items = pm.load_all(mgr)
        for item in current_test_mapping().items():
            self.assertIn(item, all_items)