import os
import re
import sys
from importlib import util as importlib_util
import breezy
from . import debug, errors, osutils, trace
def _env_plugin_path(key='BRZ_PLUGIN_PATH'):
    """Gives list of paths and contexts for plugins from environ key.

    Each entry is either a specific path to load plugins from and the value
    'path', or None and one of the three values 'user', 'core', 'site'.
    """
    path_details = []
    env = os.environ.get(key)
    defaults = {'user': not env, 'core': True, 'site': True}
    if env:
        for p in env.split(os.pathsep):
            flag, name = (p[:1], p[1:])
            if flag in ('+', '-') and name in defaults:
                if flag == '+' and defaults[name] is not None:
                    path_details.append((None, name))
                defaults[name] = None
            else:
                path_details.append((p, 'path'))
    for name in ('user', 'core', 'site'):
        if defaults[name]:
            path_details.append((None, name))
    return path_details