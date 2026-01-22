import logging
import re
from argparse import (
from collections import defaultdict
from functools import total_ordering
from itertools import starmap
from string import Template
from typing import Any, Dict, List
from typing import Optional as Opt
from typing import Union
def add_argument_to(parser: ArgumentParser, option_string: Union[str, List[str]]='--print-completion', help: str='print shell completion script', parent: Opt[ArgumentParser]=None, preamble: Union[str, Dict[str, str]]=''):
    """
    option_string:
      iff positional (no `-` prefix) then `parser` is assumed to actually be
      a subparser (subcommand mode)
    parent:
      required in subcommand mode
    """
    if isinstance(option_string, str):
        option_string = [option_string]
    kwargs = {'choices': SUPPORTED_SHELLS, 'default': None, 'help': help, 'action': completion_action(parent, preamble)}
    if option_string[0][0] != '-':
        kwargs.update(default=SUPPORTED_SHELLS[0], nargs='?')
        assert parent is not None, 'subcommand mode: parent required'
    parser.add_argument(*option_string, **kwargs)
    return parser