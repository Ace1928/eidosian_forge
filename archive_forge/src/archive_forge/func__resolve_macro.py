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
def _resolve_macro(self, statement: Statement) -> Optional[str]:
    """
        Resolve a macro and return the resulting string

        :param statement: the parsed statement from the command line
        :return: the resolved macro or None on error
        """
    if statement.command not in self.macros.keys():
        raise KeyError(f'{statement.command} is not a macro')
    macro = self.macros[statement.command]
    if len(statement.arg_list) < macro.minimum_arg_count:
        plural = '' if macro.minimum_arg_count == 1 else 's'
        self.perror(f"The macro '{statement.command}' expects at least {macro.minimum_arg_count} argument{plural}")
        return None
    resolved = macro.value
    reverse_arg_list = sorted(macro.arg_list, key=lambda ma: ma.start_index, reverse=True)
    for macro_arg in reverse_arg_list:
        if macro_arg.is_escaped:
            to_replace = '{{' + macro_arg.number_str + '}}'
            replacement = '{' + macro_arg.number_str + '}'
        else:
            to_replace = '{' + macro_arg.number_str + '}'
            replacement = statement.argv[int(macro_arg.number_str)]
        parts = resolved.rsplit(to_replace, maxsplit=1)
        resolved = parts[0] + replacement + parts[1]
    for stmt_arg in statement.arg_list[macro.minimum_arg_count:]:
        resolved += ' ' + stmt_arg
    return resolved + statement.post_command