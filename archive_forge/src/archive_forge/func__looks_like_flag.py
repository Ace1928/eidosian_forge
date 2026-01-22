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
def _looks_like_flag(token: str, parser: argparse.ArgumentParser) -> bool:
    """
    Determine if a token looks like a flag. Unless an argument has nargs set to argparse.REMAINDER,
    then anything that looks like a flag can't be consumed as a value for it.
    Based on argparse._parse_optional().
    """
    if len(token) < 2:
        return False
    if not token[0] in parser.prefix_chars:
        return False
    if parser._negative_number_matcher.match(token):
        if not parser._has_negative_number_optionals:
            return False
    if ' ' in token:
        return False
    return True