from the CompletionItem.description value.
import argparse
import re
import sys
from argparse import (
from gettext import (
from typing import (
from . import (
def _match_argument_wrapper(self: argparse.ArgumentParser, action: argparse.Action, arg_strings_pattern: str) -> int:
    nargs_pattern = self._get_nargs_pattern(action)
    match = re.match(nargs_pattern, arg_strings_pattern)
    if match is None:
        nargs_range = action.get_nargs_range()
        if nargs_range is not None:
            raise ArgumentError(action, generate_range_error(nargs_range[0], nargs_range[1]))
    return orig_argument_parser_match_argument(self, action, arg_strings_pattern)