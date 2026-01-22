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
def _input_line_to_statement(self, line: str) -> Statement:
    """
        Parse the user's input line and convert it to a Statement, ensuring that all macros are also resolved

        :param line: the line being parsed
        :return: parsed command line as a Statement
        :raises: Cmd2ShlexError if a shlex error occurs (e.g. No closing quotation)
        :raises: EmptyStatement when the resulting Statement is blank
        """
    used_macros = []
    orig_line = None
    while True:
        statement = self._complete_statement(line)
        if orig_line is None:
            orig_line = statement.raw
        if statement.command in self.macros.keys() and statement.command not in used_macros:
            used_macros.append(statement.command)
            resolve_result = self._resolve_macro(statement)
            if resolve_result is None:
                raise EmptyStatement
            line = resolve_result
        else:
            break
    if orig_line != statement.raw:
        statement = Statement(statement.args, raw=orig_line, command=statement.command, arg_list=statement.arg_list, multiline_command=statement.multiline_command, terminator=statement.terminator, suffix=statement.suffix, pipe_to=statement.pipe_to, output=statement.output, output_to=statement.output_to)
    return statement