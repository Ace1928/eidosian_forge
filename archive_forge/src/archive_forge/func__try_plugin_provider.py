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
def _try_plugin_provider(cmd_name):
    """Probe for a plugin provider having cmd_name."""
    try:
        plugin_metadata, provider = probe_for_provider(cmd_name)
        raise CommandAvailableInPlugin(cmd_name, plugin_metadata, provider)
    except NoPluginAvailable:
        pass