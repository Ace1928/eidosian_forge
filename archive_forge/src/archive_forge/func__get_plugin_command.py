import contextlib
import os
import sys
from typing import List, Optional, Type, Union
from . import i18n, option, osutils, trace
from .lazy_import import lazy_import
import breezy
from breezy import (
from . import errors, registry
from .hooks import Hooks
from .i18n import gettext
from .plugin import disable_plugins, load_plugins, plugin_name
def _get_plugin_command(cmd_or_None, cmd_name):
    """Get a command from brz's plugins."""
    try:
        return plugin_cmds.get(cmd_name)()
    except KeyError:
        pass
    for key in plugin_cmds.keys():
        info = plugin_cmds.get_info(key)
        if cmd_name in info.aliases:
            return plugin_cmds.get(key)()
    return cmd_or_None