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
@as_subcommand_to('alias', 'create', alias_create_parser, help=alias_create_description.lower())
def _alias_create(self, args: argparse.Namespace) -> None:
    """Create or overwrite an alias"""
    self.last_result = False
    valid, errmsg = self.statement_parser.is_valid_command(args.name)
    if not valid:
        self.perror(f'Invalid alias name: {errmsg}')
        return
    if args.name in self.get_all_commands():
        self.perror('Alias cannot have the same name as a command')
        return
    if args.name in self.macros:
        self.perror('Alias cannot have the same name as a macro')
        return
    tokens_to_unquote = constants.REDIRECTION_TOKENS
    tokens_to_unquote.extend(self.statement_parser.terminators)
    utils.unquote_specific_tokens(args.command_args, tokens_to_unquote)
    value = args.command
    if args.command_args:
        value += ' ' + ' '.join(args.command_args)
    result = 'overwritten' if args.name in self.aliases else 'created'
    self.poutput(f"Alias '{args.name}' {result}")
    self.aliases[args.name] = value
    self.last_result = True