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
def _set_up_cmd2_readline(self) -> _SavedReadlineSettings:
    """
        Called at beginning of command loop to set up readline with cmd2-specific settings

        :return: Class containing saved readline settings
        """
    readline_settings = _SavedReadlineSettings()
    if self._completion_supported():
        if rl_type == RlType.GNU:
            readline_settings.basic_quotes = cast(bytes, ctypes.cast(rl_basic_quote_characters, ctypes.c_void_p).value)
            rl_basic_quote_characters.value = None
        readline_settings.completer = readline.get_completer()
        readline.set_completer(self.complete)
        completer_delims = ' \t\n'
        completer_delims += ''.join(constants.QUOTES)
        completer_delims += ''.join(constants.REDIRECTION_CHARS)
        completer_delims += ''.join(self.statement_parser.terminators)
        readline_settings.delims = readline.get_completer_delims()
        readline.set_completer_delims(completer_delims)
        readline.parse_and_bind(self.completekey + ': complete')
    return readline_settings