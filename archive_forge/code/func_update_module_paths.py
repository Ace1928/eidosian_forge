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
def update_module_paths(self, paths):
    paths = plugin.extend_path(paths, self.module_name)
    self.module.__path__ = paths
    self.log('using %r', paths)
    return paths