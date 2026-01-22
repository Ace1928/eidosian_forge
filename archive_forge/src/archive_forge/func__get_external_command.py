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
def _get_external_command(cmd_or_None, cmd_name):
    """Lookup a command that is a shell script."""
    if cmd_or_None is not None:
        return cmd_or_None
    from breezy.externalcommand import ExternalCommand
    cmd_obj = ExternalCommand.find_command(cmd_name)
    if cmd_obj:
        return cmd_obj