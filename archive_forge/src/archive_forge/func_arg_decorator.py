import argparse
from typing import (
from . import (
from .argparse_custom import (
from .command_definition import (
from .exceptions import (
from .parsing import (
from .utils import (
def arg_decorator(func: ArgparseCommandFunc) -> ArgparseCommandFunc:
    _set_parser_prog(parser, command + ' ' + subcommand)
    if parser.description is None and func.__doc__:
        parser.description = func.__doc__
    setattr(func, constants.SUBCMD_ATTR_COMMAND, command)
    setattr(func, constants.CMD_ATTR_ARGPARSER, parser)
    setattr(func, constants.SUBCMD_ATTR_NAME, subcommand)
    add_parser_kwargs: Dict[str, Any] = dict()
    if help is not None:
        add_parser_kwargs['help'] = help
    if aliases:
        add_parser_kwargs['aliases'] = aliases[:]
    setattr(func, constants.SUBCMD_ATTR_ADD_PARSER_KWARGS, add_parser_kwargs)
    return func