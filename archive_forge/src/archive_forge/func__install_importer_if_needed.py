import os
import re
import sys
from importlib import util as importlib_util
import breezy
from . import debug, errors, osutils, trace
def _install_importer_if_needed(plugin_details):
    """Install a meta path finder to handle plugin_details if any."""
    if plugin_details:
        finder = _PluginsAtFinder(_MODULE_PREFIX, plugin_details)
        sys.meta_path.insert(2, finder)