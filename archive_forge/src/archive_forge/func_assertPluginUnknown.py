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
def assertPluginUnknown(self, name):
    self.assertTrue(getattr(self.module, name, None) is None)
    self.assertFalse(self.module_prefix + name in sys.modules)