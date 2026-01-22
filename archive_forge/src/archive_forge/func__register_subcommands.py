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
def _register_subcommands(self, cmdset: Union[CommandSet, 'Cmd']) -> None:
    """
        Register subcommands with their base command

        :param cmdset: CommandSet or cmd2.Cmd subclass containing subcommands
        """
    if not (cmdset is self or cmdset in self._installed_command_sets):
        raise CommandSetRegistrationError('Cannot register subcommands with an unregistered CommandSet')
    methods = inspect.getmembers(cmdset, predicate=lambda meth: isinstance(meth, Callable) and hasattr(meth, constants.SUBCMD_ATTR_NAME) and hasattr(meth, constants.SUBCMD_ATTR_COMMAND) and hasattr(meth, constants.CMD_ATTR_ARGPARSER))
    for method_name, method in methods:
        subcommand_name: str = getattr(method, constants.SUBCMD_ATTR_NAME)
        full_command_name: str = getattr(method, constants.SUBCMD_ATTR_COMMAND)
        subcmd_parser = getattr(method, constants.CMD_ATTR_ARGPARSER)
        subcommand_valid, errmsg = self.statement_parser.is_valid_command(subcommand_name, is_subcommand=True)
        if not subcommand_valid:
            raise CommandSetRegistrationError(f'Subcommand {str(subcommand_name)} is not valid: {errmsg}')
        command_tokens = full_command_name.split()
        command_name = command_tokens[0]
        subcommand_names = command_tokens[1:]
        if command_name in self.disabled_commands:
            command_func = self.disabled_commands[command_name].command_function
        else:
            command_func = self.cmd_func(command_name)
        if command_func is None:
            raise CommandSetRegistrationError(f"Could not find command '{command_name}' needed by subcommand: {str(method)}")
        command_parser = getattr(command_func, constants.CMD_ATTR_ARGPARSER, None)
        if command_parser is None:
            raise CommandSetRegistrationError(f"Could not find argparser for command '{command_name}' needed by subcommand: {str(method)}")

        def find_subcommand(action: argparse.ArgumentParser, subcmd_names: List[str]) -> argparse.ArgumentParser:
            if not subcmd_names:
                return action
            cur_subcmd = subcmd_names.pop(0)
            for sub_action in action._actions:
                if isinstance(sub_action, argparse._SubParsersAction):
                    for choice_name, choice in sub_action.choices.items():
                        if choice_name == cur_subcmd:
                            return find_subcommand(choice, subcmd_names)
                    break
            raise CommandSetRegistrationError(f"Could not find subcommand '{full_command_name}'")
        target_parser = find_subcommand(command_parser, subcommand_names)
        for action in target_parser._actions:
            if isinstance(action, argparse._SubParsersAction):
                action.remove_parser(subcommand_name)
                add_parser_kwargs = getattr(method, constants.SUBCMD_ATTR_ADD_PARSER_KWARGS, {})
                add_parser_kwargs['parents'] = [subcmd_parser]
                add_parser_kwargs['prog'] = subcmd_parser.prog
                add_parser_kwargs['usage'] = subcmd_parser.usage
                add_parser_kwargs['description'] = subcmd_parser.description
                add_parser_kwargs['epilog'] = subcmd_parser.epilog
                add_parser_kwargs['formatter_class'] = subcmd_parser.formatter_class
                add_parser_kwargs['prefix_chars'] = subcmd_parser.prefix_chars
                add_parser_kwargs['fromfile_prefix_chars'] = subcmd_parser.fromfile_prefix_chars
                add_parser_kwargs['argument_default'] = subcmd_parser.argument_default
                add_parser_kwargs['conflict_handler'] = subcmd_parser.conflict_handler
                add_parser_kwargs['allow_abbrev'] = subcmd_parser.allow_abbrev
                add_parser_kwargs['add_help'] = False
                attached_parser = action.add_parser(subcommand_name, **add_parser_kwargs)
                defaults = {constants.NS_ATTR_SUBCMD_HANDLER: method}
                attached_parser.set_defaults(**defaults)
                attached_parser.set_ap_completer_type(subcmd_parser.get_ap_completer_type())
                setattr(attached_parser, constants.PARSER_ATTR_COMMANDSET, cmdset)
                break