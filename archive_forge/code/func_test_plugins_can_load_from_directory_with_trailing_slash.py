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
def test_plugins_can_load_from_directory_with_trailing_slash(self):
    self.assertPluginUnknown('ts_plugin')
    tempattribute = 'trailing-slash'
    self.assertFalse(tempattribute in self.activeattributes)
    breezy.tests.test_plugins.TestLoadingPlugins.activeattributes[tempattribute] = []
    self.assertTrue(tempattribute in self.activeattributes)
    os.mkdir('plugin_test')
    template = "from breezy.tests.test_plugins import TestLoadingPlugins\nTestLoadingPlugins.activeattributes[%r].append('%s')\n"
    with open(os.path.join('plugin_test', 'ts_plugin.py'), 'w') as outfile:
        outfile.write(template % (tempattribute, 'plugin'))
        outfile.write('\n')
    try:
        self.load_with_paths(['plugin_test' + os.sep])
        self.assertEqual(['plugin'], self.activeattributes[tempattribute])
        self.assertPluginKnown('ts_plugin')
    finally:
        del self.activeattributes[tempattribute]