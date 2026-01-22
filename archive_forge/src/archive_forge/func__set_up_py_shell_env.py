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
def _set_up_py_shell_env(self, interp: InteractiveConsole) -> _SavedCmd2Env:
    """
        Set up interactive Python shell environment
        :return: Class containing saved up cmd2 environment
        """
    cmd2_env = _SavedCmd2Env()
    if rl_type != RlType.NONE:
        for i in range(1, readline.get_current_history_length() + 1):
            cmd2_env.history.append(readline.get_history_item(i))
        readline.clear_history()
        for item in self._py_history:
            readline.add_history(item)
        if self._completion_supported():
            if rl_type == RlType.GNU:
                cmd2_env.readline_settings.basic_quotes = cast(bytes, ctypes.cast(rl_basic_quote_characters, ctypes.c_void_p).value)
                rl_basic_quote_characters.value = orig_rl_basic_quotes
                if 'gnureadline' in sys.modules:
                    if 'readline' in sys.modules:
                        cmd2_env.readline_module = sys.modules['readline']
                    sys.modules['readline'] = sys.modules['gnureadline']
            cmd2_env.readline_settings.delims = readline.get_completer_delims()
            readline.set_completer_delims(orig_rl_delims)
            if rl_type == RlType.GNU:
                readline.set_completion_display_matches_hook(None)
            elif rl_type == RlType.PYREADLINE:
                readline.rl.mode._display_completions = orig_pyreadline_display
            cmd2_env.readline_settings.completer = readline.get_completer()
            interp.runcode('from rlcompleter import Completer')
            interp.runcode('import readline')
            interp.runcode('readline.set_completer(Completer(locals()).complete)')
    self._reset_py_display()
    cmd2_env.sys_stdout = sys.stdout
    sys.stdout = self.stdout
    cmd2_env.sys_stdin = sys.stdin
    sys.stdin = self.stdin
    return cmd2_env