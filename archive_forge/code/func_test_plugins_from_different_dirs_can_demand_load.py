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
def test_plugins_from_different_dirs_can_demand_load(self):
    self.assertFalse('breezy.plugins.pluginone' in sys.modules)
    self.assertFalse('breezy.plugins.plugintwo' in sys.modules)
    tempattribute = 'different-dirs'
    self.assertFalse(tempattribute in self.activeattributes)
    breezy.tests.test_plugins.TestLoadingPlugins.activeattributes[tempattribute] = []
    self.assertTrue(tempattribute in self.activeattributes)
    os.mkdir('first')
    os.mkdir('second')
    template = "from breezy.tests.test_plugins import TestLoadingPlugins\nTestLoadingPlugins.activeattributes[%r].append('%s')\n"
    with open(os.path.join('first', 'pluginone.py'), 'w') as outfile:
        outfile.write(template % (tempattribute, 'first'))
        outfile.write('\n')
    with open(os.path.join('second', 'plugintwo.py'), 'w') as outfile:
        outfile.write(template % (tempattribute, 'second'))
        outfile.write('\n')
    try:
        self.assertPluginUnknown('pluginone')
        self.assertPluginUnknown('plugintwo')
        self.update_module_paths(['first', 'second'])
        exec('import %spluginone' % self.module_prefix)
        self.assertEqual(['first'], self.activeattributes[tempattribute])
        exec('import %splugintwo' % self.module_prefix)
        self.assertEqual(['first', 'second'], self.activeattributes[tempattribute])
    finally:
        del self.activeattributes[tempattribute]