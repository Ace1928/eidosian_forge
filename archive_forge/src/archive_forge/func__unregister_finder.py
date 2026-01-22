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
def _unregister_finder(self):
    """Removes any test copies of _PluginsAtFinder from sys.meta_path."""
    idx = len(sys.meta_path)
    while idx:
        idx -= 1
        finder = sys.meta_path[idx]
        if getattr(finder, 'prefix', '') == self.module_prefix:
            self.log('removed %r from sys.meta_path', finder)
            sys.meta_path.pop(idx)