import argparse
import inspect
import numbers
from collections import (
from typing import (
from .ansi import (
from .constants import (
from .argparse_custom import (
from .command_definition import (
from .exceptions import (
from .table_creator import (
def update_mutex_groups(arg_action: argparse.Action) -> None:
    """
            Check if an argument belongs to a mutually exclusive group and either mark that group
            as complete or print an error if the group has already been completed
            :param arg_action: the action of the argument
            :raises: CompletionError if the group is already completed
            """
    for group in self._parser._mutually_exclusive_groups:
        if arg_action in group._group_actions:
            if group in completed_mutex_groups:
                completer_action = completed_mutex_groups[group]
                if arg_action == completer_action:
                    return
                error = 'Error: argument {}: not allowed with argument {}'.format(argparse._get_action_name(arg_action), argparse._get_action_name(completer_action))
                raise CompletionError(error)
            completed_mutex_groups[group] = arg_action
            for group_action in group._group_actions:
                if group_action == arg_action:
                    continue
                elif group_action in self._flag_to_action.values():
                    matched_flags.extend(group_action.option_strings)
                elif group_action in remaining_positionals:
                    remaining_positionals.remove(group_action)
            break