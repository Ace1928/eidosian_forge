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
class CommandAvailableInPlugin(Exception):
    internal_error = False

    def __init__(self, cmd_name, plugin_metadata, provider):
        self.plugin_metadata = plugin_metadata
        self.cmd_name = cmd_name
        self.provider = provider

    def __str__(self):
        _fmt = '"%s" is not a standard brz command. \nHowever, the following official plugin provides this command: %s\nYou can install it by going to: %s' % (self.cmd_name, self.plugin_metadata['name'], self.plugin_metadata['url'])
        return _fmt