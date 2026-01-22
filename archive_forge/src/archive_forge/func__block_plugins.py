import os
import re
import sys
from importlib import util as importlib_util
import breezy
from . import debug, errors, osutils, trace
def _block_plugins(names):
    """Add names to sys.modules to block future imports."""
    for name in names:
        package_name = _MODULE_PREFIX + name
        if sys.modules.get(package_name) is not None:
            trace.mutter('Blocked plugin %s already loaded.', name)
        sys.modules[package_name] = None