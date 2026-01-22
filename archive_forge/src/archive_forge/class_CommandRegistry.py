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
class CommandRegistry(registry.Registry):
    """Special registry mapping command names to command classes.

    Attributes:
      overridden_registry: Look in this registry for commands being
        overridden by this registry.  This can be used to tell plugin commands
        about the builtin they're decorating.
    """

    def __init__(self):
        registry.Registry.__init__(self)
        self.overridden_registry = None
        self._alias_dict = {}

    def get(self, command_name):
        real_name = self._alias_dict.get(command_name, command_name)
        return registry.Registry.get(self, real_name)

    @staticmethod
    def _get_name(command_name):
        if command_name.startswith('cmd_'):
            return _unsquish_command_name(command_name)
        else:
            return command_name

    def register(self, cmd, decorate=False):
        """Utility function to help register a command

        Args:
          cmd: Command subclass to register
          decorate: If true, allow overriding an existing command
            of the same name; the old command is returned by this function.
            Otherwise it is an error to try to override an existing command.
        """
        k = cmd.__name__
        k_unsquished = self._get_name(k)
        try:
            previous = self.get(k_unsquished)
        except KeyError:
            previous = None
            if self.overridden_registry:
                try:
                    previous = self.overridden_registry.get(k_unsquished)
                except KeyError:
                    pass
        info = CommandInfo.from_command(cmd)
        try:
            registry.Registry.register(self, k_unsquished, cmd, override_existing=decorate, info=info)
        except KeyError:
            trace.warning('Two plugins defined the same command: %r' % k)
            trace.warning('Not loading the one in %r' % sys.modules[cmd.__module__])
            trace.warning('Previously this command was registered from %r' % sys.modules[previous.__module__])
        for a in cmd.aliases:
            self._alias_dict[a] = k_unsquished
        return previous

    def register_lazy(self, command_name, aliases, module_name):
        """Register a command without loading its module.

        Args:
          command_name: The primary name of the command.
          aliases: A list of aliases for the command.
          module_name: The module that the command lives in.
        """
        key = self._get_name(command_name)
        registry.Registry.register_lazy(self, key, module_name, command_name, info=CommandInfo(aliases))
        for a in aliases:
            self._alias_dict[a] = key