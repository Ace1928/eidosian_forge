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
def _single_prefix_char(token: str, parser: argparse.ArgumentParser) -> bool:
    """Returns if a token is just a single flag prefix character"""
    return len(token) == 1 and token[0] in parser.prefix_chars