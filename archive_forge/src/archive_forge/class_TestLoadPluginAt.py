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
class TestLoadPluginAt(BaseTestPlugins):

    def setUp(self):
        super().setUp()
        self.create_plugin_package('test_foo', dir='non-standard-dir')
        self.create_plugin_package('test_foo', dir='standard/test_foo')

    def assertTestFooLoadedFrom(self, path):
        self.assertPluginKnown('test_foo')
        self.assertDocstring('This is the doc for test_foo', self.module.test_foo)
        self.assertEqual(path, self.module.test_foo.dir_source)

    def test_regular_load(self):
        self.load_with_paths(['standard'])
        self.assertTestFooLoadedFrom('standard/test_foo')

    def test_import(self):
        self.overrideEnv('BRZ_PLUGINS_AT', 'test_foo@non-standard-dir')
        self.update_module_paths(['standard'])
        import breezy.testingplugins.test_foo
        self.assertTestFooLoadedFrom('non-standard-dir')

    def test_loading(self):
        self.overrideEnv('BRZ_PLUGINS_AT', 'test_foo@non-standard-dir')
        self.load_with_paths(['standard'])
        self.assertTestFooLoadedFrom('non-standard-dir')

    def test_loading_other_name(self):
        self.overrideEnv('BRZ_PLUGINS_AT', 'test_foo@non-standard-dir')
        os.rename('standard/test_foo', 'standard/test_bar')
        self.load_with_paths(['standard'])
        self.assertTestFooLoadedFrom('non-standard-dir')

    def test_compiled_loaded(self):
        self.overrideEnv('BRZ_PLUGINS_AT', 'test_foo@non-standard-dir')
        self.load_with_paths(['standard'])
        self.assertTestFooLoadedFrom('non-standard-dir')
        self.assertIsSameRealPath('non-standard-dir/__init__.py', self.module.test_foo.__file__)
        os.remove('non-standard-dir/__init__.py')
        self.promote_cache('non-standard-dir')
        self.reset()
        self.load_with_paths(['standard'])
        self.assertTestFooLoadedFrom('non-standard-dir')
        suffix = plugin.COMPILED_EXT
        self.assertIsSameRealPath('non-standard-dir/__init__' + suffix, self.module.test_foo.__file__)

    def test_submodule_loading(self):
        self.create_plugin_package('test_bar', dir='non-standard-dir/test_bar')
        self.overrideEnv('BRZ_PLUGINS_AT', 'test_foo@non-standard-dir')
        self.update_module_paths(['standard'])
        import breezy.testingplugins.test_foo
        self.assertEqual(self.module_prefix + 'test_foo', self.module.test_foo.__package__)
        import breezy.testingplugins.test_foo.test_bar
        self.assertIsSameRealPath('non-standard-dir/test_bar/__init__.py', self.module.test_foo.test_bar.__file__)

    def test_relative_submodule_loading(self):
        self.create_plugin_package('test_foo', dir='another-dir', source='\nfrom . import test_bar\n')
        self.create_plugin_package('test_bar', dir='another-dir/test_bar')
        self.overrideEnv('BRZ_PLUGINS_AT', 'test_foo@another-dir')
        self.update_module_paths(['standard'])
        import breezy.testingplugins.test_foo
        self.assertEqual(self.module_prefix + 'test_foo', self.module.test_foo.__package__)
        self.assertIsSameRealPath('another-dir/test_bar/__init__.py', self.module.test_foo.test_bar.__file__)

    def test_loading_from___init__only(self):
        init = 'non-standard-dir/__init__.py'
        random = 'non-standard-dir/setup.py'
        os.rename(init, random)
        self.overrideEnv('BRZ_PLUGINS_AT', 'test_foo@non-standard-dir')
        self.load_with_paths(['standard'])
        self.assertPluginUnknown('test_foo')

    def test_loading_from_specific_file(self):
        plugin_dir = 'non-standard-dir'
        plugin_file_name = 'iamtestfoo.py'
        plugin_path = osutils.pathjoin(plugin_dir, plugin_file_name)
        source = '"""This is the doc for {}"""\ndir_source = \'{}\'\n'.format('test_foo', plugin_path)
        self.create_plugin('test_foo', source=source, dir=plugin_dir, file_name=plugin_file_name)
        self.overrideEnv('BRZ_PLUGINS_AT', 'test_foo@%s' % plugin_path)
        self.load_with_paths(['standard'])
        self.assertTestFooLoadedFrom(plugin_path)