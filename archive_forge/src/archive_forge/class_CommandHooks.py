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
class CommandHooks(Hooks):
    """Hooks related to Command object creation/enumeration."""

    def __init__(self):
        """Create the default hooks.

        These are all empty initially, because by default nothing should get
        notified.
        """
        Hooks.__init__(self, 'breezy.commands', 'Command.hooks')
        self.add_hook('extend_command', 'Called after creating a command object to allow modifications such as adding or removing options, docs etc. Called with the new breezy.commands.Command object.', (1, 13))
        self.add_hook('get_command', 'Called when creating a single command. Called with (cmd_or_None, command_name). get_command should either return the cmd_or_None parameter, or a replacement Command object that should be used for the command. Note that the Command.hooks hooks are core infrastructure. Many users will prefer to use breezy.commands.register_command or plugin_cmds.register_lazy.', (1, 17))
        self.add_hook('get_missing_command', 'Called when creating a single command if no command could be found. Called with (command_name). get_missing_command should either return None, or a Command object to be used for the command.', (1, 17))
        self.add_hook('list_commands', 'Called when enumerating commands. Called with a set of cmd_name strings for all the commands found so far. This set  is safe to mutate - e.g. to remove a command. list_commands should return the updated set of command names.', (1, 17))
        self.add_hook('pre_command', 'Called prior to executing a command. Called with the command object.', (2, 6))
        self.add_hook('post_command', 'Called after executing a command. Called with the command object.', (2, 6))