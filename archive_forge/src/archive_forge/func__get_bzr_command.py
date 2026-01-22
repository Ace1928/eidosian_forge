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
def _get_bzr_command(cmd_or_None, cmd_name):
    """Get a command from bzr's core."""
    try:
        cmd_class = builtin_command_registry.get(cmd_name)
    except KeyError:
        pass
    else:
        return cmd_class()
    return cmd_or_None