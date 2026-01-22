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
class _UnfinishedFlagError(CompletionError):

    def __init__(self, flag_arg_state: _ArgumentState) -> None:
        """
        CompletionError which occurs when the user has not finished the current flag
        :param flag_arg_state: information about the unfinished flag action
        """
        error = 'Error: argument {}: {} ({} entered)'.format(argparse._get_action_name(flag_arg_state.action), generate_range_error(cast(int, flag_arg_state.min), cast(Union[int, float], flag_arg_state.max)), flag_arg_state.count)
        super().__init__(error)