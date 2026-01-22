import importlib
import logging
import os
import sys
import types
from io import StringIO
from typing import Any, Dict, List
import breezy
from .. import osutils, plugin, tests
from . import test_bar
class TestPlugins(BaseTestPlugins):

    def setup_plugin(self, source=''):
        self.assertPluginUnknown('plugin')
        with open('plugin.py', 'w') as f:
            f.write(source + '\n')
        self.load_with_paths(['.'])

    def test_plugin_loaded(self):
        self.assertPluginUnknown('plugin')
        self.assertIs(None, breezy.plugin.get_loaded_plugin('plugin'))
        self.setup_plugin()
        p = breezy.plugin.get_loaded_plugin('plugin')
        self.assertIsInstance(p, breezy.plugin.PlugIn)
        self.assertIs(p.module, sys.modules[self.module_prefix + 'plugin'])

    def test_plugin_loaded_disabled(self):
        self.assertPluginUnknown('plugin')
        self.overrideEnv('BRZ_DISABLE_PLUGINS', 'plugin')
        self.setup_plugin()
        self.assertIs(None, breezy.plugin.get_loaded_plugin('plugin'))

    def test_plugin_appears_in_plugins(self):
        self.setup_plugin()
        self.assertPluginKnown('plugin')
        p = self.plugins['plugin']
        self.assertIsInstance(p, breezy.plugin.PlugIn)
        self.assertIs(p.module, sys.modules[self.module_prefix + 'plugin'])

    def test_trivial_plugin_get_path(self):
        self.setup_plugin()
        p = self.plugins['plugin']
        plugin_path = self.test_dir + '/plugin.py'
        self.assertIsSameRealPath(plugin_path, osutils.normpath(p.path()))

    def test_plugin_get_path_py_not_pyc(self):
        self.setup_plugin()
        self.promote_cache(self.test_dir)
        self.reset()
        self.load_with_paths(['.'])
        p = plugin.plugins()['plugin']
        plugin_path = self.test_dir + '/plugin.py'
        self.assertIsSameRealPath(plugin_path, osutils.normpath(p.path()))

    def test_plugin_get_path_pyc_only(self):
        self.setup_plugin()
        os.unlink(self.test_dir + '/plugin.py')
        self.promote_cache(self.test_dir)
        self.reset()
        self.load_with_paths(['.'])
        p = plugin.plugins()['plugin']
        plugin_path = self.test_dir + '/plugin' + plugin.COMPILED_EXT
        self.assertIsSameRealPath(plugin_path, osutils.normpath(p.path()))

    def test_no_test_suite_gives_None_for_test_suite(self):
        self.setup_plugin()
        p = plugin.plugins()['plugin']
        self.assertEqual(None, p.test_suite())

    def test_test_suite_gives_test_suite_result(self):
        source = "def test_suite(): return 'foo'"
        self.setup_plugin(source)
        p = plugin.plugins()['plugin']
        self.assertEqual('foo', p.test_suite())

    def test_no_load_plugin_tests_gives_None_for_load_plugin_tests(self):
        self.setup_plugin()
        loader = tests.TestUtil.TestLoader()
        p = plugin.plugins()['plugin']
        self.assertEqual(None, p.load_plugin_tests(loader))

    def test_load_plugin_tests_gives_load_plugin_tests_result(self):
        source = "\ndef load_tests(loader, standard_tests, pattern):\n    return 'foo'"
        self.setup_plugin(source)
        loader = tests.TestUtil.TestLoader()
        p = plugin.plugins()['plugin']
        self.assertEqual('foo', p.load_plugin_tests(loader))

    def check_version_info(self, expected, source='', name='plugin'):
        self.setup_plugin(source)
        self.assertEqual(expected, plugin.plugins()[name].version_info())

    def test_no_version_info(self):
        self.check_version_info(None)

    def test_with_version_info(self):
        self.check_version_info((1, 2, 3, 'dev', 4), "version_info = (1, 2, 3, 'dev', 4)")

    def test_short_version_info_gets_padded(self):
        self.check_version_info((1, 2, 3, 'final', 0), 'version_info = (1, 2, 3)')

    def check_version(self, expected, source=None, name='plugin'):
        self.setup_plugin(source)
        self.assertEqual(expected, plugins[name].__version__)

    def test_no_version_info___version__(self):
        self.setup_plugin()
        plugin = breezy.plugin.plugins()['plugin']
        self.assertEqual('unknown', plugin.__version__)

    def test_str__version__with_version_info(self):
        self.setup_plugin("version_info = '1.2.3'")
        plugin = breezy.plugin.plugins()['plugin']
        self.assertEqual('1.2.3', plugin.__version__)

    def test_noniterable__version__with_version_info(self):
        self.setup_plugin('version_info = (1)')
        plugin = breezy.plugin.plugins()['plugin']
        self.assertEqual('1', plugin.__version__)

    def test_1__version__with_version_info(self):
        self.setup_plugin('version_info = (1,)')
        plugin = breezy.plugin.plugins()['plugin']
        self.assertEqual('1', plugin.__version__)

    def test_1_2__version__with_version_info(self):
        self.setup_plugin('version_info = (1, 2)')
        plugin = breezy.plugin.plugins()['plugin']
        self.assertEqual('1.2', plugin.__version__)

    def test_1_2_3__version__with_version_info(self):
        self.setup_plugin('version_info = (1, 2, 3)')
        plugin = breezy.plugin.plugins()['plugin']
        self.assertEqual('1.2.3', plugin.__version__)

    def test_candidate__version__with_version_info(self):
        self.setup_plugin("version_info = (1, 2, 3, 'candidate', 1)")
        plugin = breezy.plugin.plugins()['plugin']
        self.assertEqual('1.2.3.rc1', plugin.__version__)

    def test_dev__version__with_version_info(self):
        self.setup_plugin("version_info = (1, 2, 3, 'dev', 0)")
        plugin = breezy.plugin.plugins()['plugin']
        self.assertEqual('1.2.3.dev', plugin.__version__)

    def test_dev_fallback__version__with_version_info(self):
        self.setup_plugin("version_info = (1, 2, 3, 'dev', 4)")
        plugin = breezy.plugin.plugins()['plugin']
        self.assertEqual('1.2.3.dev4', plugin.__version__)

    def test_final__version__with_version_info(self):
        self.setup_plugin("version_info = (1, 2, 3, 'final', 0)")
        plugin = breezy.plugin.plugins()['plugin']
        self.assertEqual('1.2.3', plugin.__version__)

    def test_final_fallback__version__with_version_info(self):
        self.setup_plugin("version_info = (1, 2, 3, 'final', 2)")
        plugin = breezy.plugin.plugins()['plugin']
        self.assertEqual('1.2.3.2', plugin.__version__)