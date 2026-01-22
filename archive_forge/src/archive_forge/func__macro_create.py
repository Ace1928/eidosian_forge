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
@as_subcommand_to('macro', 'create', macro_create_parser, help=macro_create_help)
def _macro_create(self, args: argparse.Namespace) -> None:
    """Create or overwrite a macro"""
    self.last_result = False
    valid, errmsg = self.statement_parser.is_valid_command(args.name)
    if not valid:
        self.perror(f'Invalid macro name: {errmsg}')
        return
    if args.name in self.get_all_commands():
        self.perror('Macro cannot have the same name as a command')
        return
    if args.name in self.aliases:
        self.perror('Macro cannot have the same name as an alias')
        return
    tokens_to_unquote = constants.REDIRECTION_TOKENS
    tokens_to_unquote.extend(self.statement_parser.terminators)
    utils.unquote_specific_tokens(args.command_args, tokens_to_unquote)
    value = args.command
    if args.command_args:
        value += ' ' + ' '.join(args.command_args)
    arg_list = []
    normal_matches = re.finditer(MacroArg.macro_normal_arg_pattern, value)
    max_arg_num = 0
    arg_nums = set()
    while True:
        try:
            cur_match = normal_matches.__next__()
            cur_num_str = re.findall(MacroArg.digit_pattern, cur_match.group())[0]
            cur_num = int(cur_num_str)
            if cur_num < 1:
                self.perror('Argument numbers must be greater than 0')
                return
            arg_nums.add(cur_num)
            if cur_num > max_arg_num:
                max_arg_num = cur_num
            arg_list.append(MacroArg(start_index=cur_match.start(), number_str=cur_num_str, is_escaped=False))
        except StopIteration:
            break
    if len(arg_nums) != max_arg_num:
        self.perror(f'Not all numbers between 1 and {max_arg_num} are present in the argument placeholders')
        return
    escaped_matches = re.finditer(MacroArg.macro_escaped_arg_pattern, value)
    while True:
        try:
            cur_match = escaped_matches.__next__()
            cur_num_str = re.findall(MacroArg.digit_pattern, cur_match.group())[0]
            arg_list.append(MacroArg(start_index=cur_match.start(), number_str=cur_num_str, is_escaped=True))
        except StopIteration:
            break
    result = 'overwritten' if args.name in self.macros else 'created'
    self.poutput(f"Macro '{args.name}' {result}")
    self.macros[args.name] = Macro(name=args.name, value=value, minimum_arg_count=max_arg_num, arg_list=arg_list)
    self.last_result = True