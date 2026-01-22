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
def _complete_statement(self, line: str) -> Statement:
    """Keep accepting lines of input until the command is complete.

        There is some pretty hacky code here to handle some quirks of
        self._read_command_line(). It returns a literal 'eof' if the input
        pipe runs out. We can't refactor it because we need to retain
        backwards compatibility with the standard library version of cmd.

        :param line: the line being parsed
        :return: the completed Statement
        :raises: Cmd2ShlexError if a shlex error occurs (e.g. No closing quotation)
        :raises: EmptyStatement when the resulting Statement is blank
        """
    while True:
        try:
            statement = self.statement_parser.parse(line)
            if statement.multiline_command and statement.terminator:
                break
            if not statement.multiline_command:
                break
        except Cmd2ShlexError:
            statement = self.statement_parser.parse_command_only(line)
            if not statement.multiline_command:
                raise
        try:
            self._at_continuation_prompt = True
            self._multiline_in_progress = line + '\n'
            nextline = self._read_command_line(self.continuation_prompt)
            if nextline == 'eof':
                nextline = '\n'
                self.poutput(nextline)
            line = f'{self._multiline_in_progress}{nextline}'
        except KeyboardInterrupt:
            self.poutput('^C')
            statement = self.statement_parser.parse('')
            break
        finally:
            self._at_continuation_prompt = False
    if not statement.command:
        raise EmptyStatement
    return statement