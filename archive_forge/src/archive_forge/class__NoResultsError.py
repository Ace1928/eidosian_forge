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
class _NoResultsError(CompletionError):

    def __init__(self, parser: argparse.ArgumentParser, arg_action: argparse.Action) -> None:
        """
        CompletionError which occurs when there are no results. If hinting is allowed, then its message will
        be a hint about the argument being tab completed.
        :param parser: ArgumentParser instance which owns the action being tab completed
        :param arg_action: action being tab completed
        """
        super().__init__(_build_hint(parser, arg_action), apply_style=False)