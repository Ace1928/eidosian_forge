import os
import re
import sys
from importlib import util as importlib_util
import breezy
from . import debug, errors, osutils, trace
def _get_site_plugin_paths(sys_paths):
    """Generate possible locations for plugins from given sys_paths."""
    for path in sys_paths:
        if os.path.basename(path) in ('dist-packages', 'site-packages'):
            yield osutils.pathjoin(path, 'breezy', 'plugins')