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
def _build_hint(parser: argparse.ArgumentParser, arg_action: argparse.Action) -> str:
    """Build tab completion hint for a given argument"""
    suppress_hint = arg_action.get_suppress_tab_hint()
    if suppress_hint or arg_action.help == argparse.SUPPRESS:
        return ''
    else:
        formatter = parser._get_formatter()
        formatter.start_section('Hint')
        formatter.add_argument(arg_action)
        formatter.end_section()
        return formatter.format_help()