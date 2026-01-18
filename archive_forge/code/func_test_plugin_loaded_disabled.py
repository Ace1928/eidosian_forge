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
def test_plugin_loaded_disabled(self):
    self.assertPluginUnknown('plugin')
    self.overrideEnv('BRZ_DISABLE_PLUGINS', 'plugin')
    self.setup_plugin()
    self.assertIs(None, breezy.plugin.get_loaded_plugin('plugin'))