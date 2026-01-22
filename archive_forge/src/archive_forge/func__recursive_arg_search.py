from __future__ import annotations
import argparse
import contextlib
import dataclasses
import difflib
import itertools
import re as _re
import shlex
import shutil
import sys
from gettext import gettext as _
from typing import Any, Dict, Generator, Iterable, List, NoReturn, Optional, Set, Tuple
from rich.columns import Columns
from rich.console import Console, Group, RenderableType
from rich.padding import Padding
from rich.panel import Panel
from rich.rule import Rule
from rich.style import Style
from rich.table import Table
from rich.text import Text
from rich.theme import Theme
from typing_extensions import override
from . import _arguments, _strings, conf
from ._parsers import ParserSpecification
def _recursive_arg_search(parser_spec: ParserSpecification, prog: str, subcommand_match_score: float) -> None:
    """Find all possible arguments that could have been passed in."""
    help_flag = ' (other subcommands) --help' if parser_spec.consolidate_subcommand_args and parser_spec.subparsers is not None else ' --help'
    for arg in parser_spec.args:
        if arg.field.is_positional() or arg.lowered.is_fixed():
            continue
        if conf.Suppress in arg.field.markers or (conf.SuppressFixed in arg.field.markers and conf.Fixed in arg.field.markers):
            continue
        option_strings = (arg.lowered.name_or_flag,)
        if arg.lowered.action is not None and callable(arg.lowered.action):
            option_strings = arg.lowered.action(option_strings, dest='').option_strings
        arguments.append(_ArgumentInfo(option_strings, metavar=arg.lowered.metavar, usage_hint=prog + help_flag, help=arg.lowered.help, subcommand_match_score=subcommand_match_score))
        nonlocal same_exists
        if not same_exists and arg.lowered.name_or_flag in unrecognized_arguments:
            same_exists = True
    if parser_spec.subparsers is not None:
        nonlocal has_subcommands
        has_subcommands = True
        for subparser_name, subparser in parser_spec.subparsers.parser_from_name.items():
            _recursive_arg_search(subparser, prog + ' ' + subparser_name, subcommand_match_score=subcommand_match_score + (1 if subparser_name in args else -0.001))
    for child in parser_spec.child_from_prefix.values():
        _recursive_arg_search(child, prog, subcommand_match_score)