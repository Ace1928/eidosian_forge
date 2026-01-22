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
def assertPluginKnown(self, name):
    self.assertTrue(getattr(self.module, name, None) is not None, 'plugins known: %r' % dir(self.module))
    self.assertTrue(self.module_prefix + name in sys.modules)