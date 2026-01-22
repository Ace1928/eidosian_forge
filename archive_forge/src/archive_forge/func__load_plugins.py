import os
import re
import sys
from importlib import util as importlib_util
import breezy
from . import debug, errors, osutils, trace
def _load_plugins(state, paths):
    """Do the importing all plugins from paths."""
    imported_names = set()
    for name, path in _iter_possible_plugins(paths):
        if name not in imported_names:
            if not valid_plugin_name(name):
                sanitised_name = sanitise_plugin_name(name)
                trace.warning("Unable to load %r in %r as a plugin because the file path isn't a valid module name; try renaming it to %r." % (name, path, sanitised_name))
                continue
            msg = _load_plugin_module(name, path)
            if msg is not None:
                state.plugin_warnings.setdefault(name, []).append(msg)
            imported_names.add(name)