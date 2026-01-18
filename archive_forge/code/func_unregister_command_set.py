import argparse
import cmd
import functools
import glob
import inspect
import os
import pydoc
import re
import sys
import threading
from code import (
from collections import (
from contextlib import (
from types import (
from typing import (
from . import (
from .argparse_custom import (
from .clipboard import (
from .command_definition import (
from .constants import (
from .decorators import (
from .exceptions import (
from .history import (
from .parsing import (
from .rl_utils import (
from .table_creator import (
from .utils import (
def unregister_command_set(self, cmdset: CommandSet) -> None:
    """
        Uninstalls a CommandSet and unloads all associated commands

        :param cmdset: CommandSet to uninstall
        """
    if cmdset in self._installed_command_sets:
        self._check_uninstallable(cmdset)
        cmdset.on_unregister()
        self._unregister_subcommands(cmdset)
        methods = inspect.getmembers(cmdset, predicate=lambda meth: isinstance(meth, Callable) and hasattr(meth, '__name__') and meth.__name__.startswith(COMMAND_FUNC_PREFIX))
        for method in methods:
            cmd_name = method[0][len(COMMAND_FUNC_PREFIX):]
            if cmd_name in self.disabled_commands:
                self.enable_command(cmd_name)
            if cmd_name in self._cmd_to_command_sets:
                del self._cmd_to_command_sets[cmd_name]
            delattr(self, COMMAND_FUNC_PREFIX + cmd_name)
            if hasattr(self, COMPLETER_FUNC_PREFIX + cmd_name):
                delattr(self, COMPLETER_FUNC_PREFIX + cmd_name)
            if hasattr(self, HELP_FUNC_PREFIX + cmd_name):
                delattr(self, HELP_FUNC_PREFIX + cmd_name)
        cmdset.on_unregistered()
        self._installed_command_sets.remove(cmdset)