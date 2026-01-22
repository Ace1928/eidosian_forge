from the CompletionItem.description value.
import argparse
import re
import sys
from argparse import (
from gettext import (
from typing import (
from . import (
def _get_nargs_pattern_wrapper(self: argparse.ArgumentParser, action: argparse.Action) -> str:
    nargs_range = action.get_nargs_range()
    if nargs_range is not None:
        if nargs_range[1] == constants.INFINITY:
            range_max = ''
        else:
            range_max = nargs_range[1]
        nargs_pattern = f'(-*A{{{nargs_range[0]},{range_max}}}-*)'
        if action.option_strings:
            nargs_pattern = nargs_pattern.replace('-*', '')
            nargs_pattern = nargs_pattern.replace('-', '')
        return nargs_pattern
    return orig_argument_parser_get_nargs_pattern(self, action)