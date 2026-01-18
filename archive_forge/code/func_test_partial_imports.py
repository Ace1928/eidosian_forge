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
def test_partial_imports(self):
    self.create_plugin('good')
    self.create_plugin('bad')
    self.create_plugin_package('ugly')
    self.overrideEnv('BRZ_DISABLE_PLUGINS', 'bad:ugly')
    self.load_with_paths(['.'])
    self.assertEqual({'good'}, self.plugins.keys())
    self.assertPluginModules({'good': self.plugins['good'].module, 'bad': None, 'ugly': None})
    self.assertNotContainsRe(self.get_log(), 'Unable to load plugin')